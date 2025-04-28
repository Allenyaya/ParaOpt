import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
import time
from datetime import timedelta
import numpy as np
import copy
import os

# 配置类定义
class Config:
    # 训练参数
    batch_size = 64
    num_epochs = 1
    max_steps = 60000  # 总迭代步数
    learning_rate = 0.001
    momentum = 0.9
    
    # 加速配置
    P = 7  # 窗口大小
    threshold = 1e-5  # 误差阈值
    ema_decay = 0.9  # 阈值指数移动平均衰减率
    adaptivity_type = 'mean'  # 自适应策略: 'mean' 或 'median'
    val_check_interval = 5  # 验证间隔(秒)
    visualize_progress = True  # 是否可视化进度
    display_time = True  # 是否显示运行时间
    
    # 系统配置
    seed = 42  # 随机种子
    device_count = torch.cuda.device_count()  # GPU数量

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)    

def to_device(batch, device):
    images, labels = batch
    return images.to(device), labels.to(device)

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def clone(self, device, other=None):
        if other is None:
            other = CNN().to(device)
            
        with torch.no_grad():
            for param_to, param_from in zip(other.parameters(), self.parameters()):
                param_to.data = param_from.data.clone()
                
        return other
        
    def get_params(self):
        return list(self.parameters())

    def set_params(self, params, clone=True):
        my_params = self.get_params()
        for p, q in zip(my_params, params):
            if clone:
                p.data = q.data.clone().to(p.device)
            else:
                p.data = q.to(p.device)
                
    def set_grads_from_grads(self, grads):
        my_params = self.get_params()
        for p, grad in zip(my_params, grads):
            if grad is not None:
                p.grad = grad.to(p.device)
                
    def compute_error_from_model(self, other):
        my_params = self.get_params()
        other_params = other.get_params()
        
        with torch.no_grad():
            error = 0.0
            total_num = 0
            for p, q in zip(my_params, other_params):
                error += torch.linalg.norm(p-q).pow(2).item()
                total_num += np.prod(list(q.shape))
        return error / total_num * 1e6

def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())

def get_all_train_iters(data_loader,config):
    dataiters = []
    for _ in range(config.max_steps):
        dataiters.append(iter(data_loader))
    return dataiters

def run(rank,total_ranks,queues,config,model,criterion,train_loader,test_loader):

    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    dataiters = get_all_train_iters(train_loader,config)
    print('Start process',rank)

    if rank == 0:
        train_loop(config, model, criterion,queues,test_loader,device)
        for _ in range(total_ranks - 1):
            queues[0].put(None)
    else:
        worker_optimizer = optim.SGD(model.parameters(), 
                                    lr=config.learning_rate, 
                                    momentum=config.momentum)
        run_worker(model,worker_optimizer,criterion,dataiters,queues,device,config.seed)

def run_worker(model, optimizer, criterion, dataiters, queues, device, seed_offset):
    
    while True:
        ret = queues[0].get()
        if ret is None:
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, criterion, dataiters, device, step, seed_offset)
        # data_iter = res['data_iter']  # 更新数据迭代器
        
        # 收集计算的梯度
        my_params = model.get_params()
        grads = [param.grad for param in my_params]
        
        # 将梯度和指标发送回主进程
        queues[1].put((grads, step, {'loss': res['loss'], 'accuracy': res['accuracy'], 
                                    'batch_size': res['batch_size'], 'correct': res['correct']}))

def take_step(model, optimizer, criterion, dataiters, device, step, seed_offset):
    """执行单个训练步骤并返回梯度"""
    # 设置随机种子以确保可重现性
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    # 获取批次数据
    batch = next(dataiters[step])
    
    images, labels = to_device(batch, device)
    
    # 前向和后向传播
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # 计算准确率
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / labels.size(0)
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        # 'data_iter': dataiters[step],
        'batch_size': labels.size(0),
        'correct': correct
    }

def train_loop(config, model, criterion, queues, test_loader, device):

    T = config.max_steps
    P = min(config.P, config.device_count)  # 调整窗口大小不超过GPU数量
    thresh = config.threshold
    
    # 初始化模型和优化器数组
    models = [None for _ in range(T+1)]
    optimizers = [None for _ in range(T+1)]
    
    # 设置初始窗口
    begin_idx, end_idx = 0, P
    total_iters = 0
    
    # 初始化统计信息
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    # 进度条初始化
    start_time = time.time()
    last_vis_time = 0
    pbar = tqdm(total=T)
    
    # 克隆初始模型到窗口中的每个位置
    for step in range(P+1):
        models[step] = model.clone(device)
        optimizers[step] = optim.SGD(models[step].parameters(), 
                                     lr=config.learning_rate, 
                                     momentum=config.momentum)
    

    while begin_idx < T:
        # 计算窗口大小
        parallel_len = end_idx - begin_idx
        
        # 存储梯度预测
        pred_f = [None for _ in range(parallel_len)]
        metrics = [None for _ in range(parallel_len)]
        
        # 分发任务到工作进程
        for i in range(parallel_len):
            step = begin_idx + i
            params = [p.data for p in models[step].get_params()]
            queues[0].put((params, step))
        
        # 收集工作进程的梯度
        for i in range(parallel_len):
            _grads, _step, _metrics = queues[1].get()
            _i = _step - begin_idx
            pred_f[_i] = _grads
            metrics[_i] = _metrics
            
            # 更新统计数据
            running_loss += _metrics['loss']
            running_correct += _metrics['correct']
            running_total += _metrics['batch_size']
        
        # 从窗口起点开始执行定点迭代
        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        
        ind = None  # 重新同步点
        errors_all = 0  # 累积误差
        
        # 对窗口中的每个位置执行定点迭代
        for i in range(parallel_len):
            step = begin_idx + i
            
            # 设置之前计算的梯度并执行优化步骤
            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            rollout_optimizer.zero_grad()
            
            # 计算与预生成模型的误差
            error = rollout_model.compute_error_from_model(models[step+1])
            
            # 基于适应性类型计算总误差
            if config.adaptivity_type == 'median':
                if i == parallel_len // 2:
                    errors_all = error
            elif config.adaptivity_type == 'mean':
                errors_all += error / parallel_len
            
            # 如果误差超过阈值或到达窗口末尾，标记同步点
            if ind is None and (error > thresh or i == parallel_len - 1):
                ind = step + 1
                optimizer_state_clone(rollout_optimizer, optimizers[step+1])
            
            # 从同步点开始克隆模型
            if ind is not None:
                models[step+1] = rollout_model.clone(device, models[step+1])
        
        # 更新阈值
        thresh = thresh * config.ema_decay + errors_all * (1 - config.ema_decay)
        
        # 滑动窗口
        new_begin_idx = ind
        new_end_idx = min(new_begin_idx + parallel_len, T)
        
        # 为新窗口区域克隆模型
        for step in range(end_idx+1, new_end_idx+1):
            models[step] = rollout_model.clone(device, 
                                              models[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None)
            optimizers[step] = optimizers[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None
        
        # 更新窗口位置和进度
        progress = new_begin_idx - begin_idx
        begin_idx = new_begin_idx
        end_idx = new_end_idx
        
        total_iters += 1
        pbar.update(progress)
        
        # 周期性输出进度
        if total_iters % 5 == 0 and running_total > 0:
            accuracy = 100 * running_correct / running_total
            avg_loss = running_loss / total_iters
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            pbar.set_description(
                f'Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {elapsed_str}'
            )
        
        # 定期验证
        # elapsed = time.time() - start_time
        # if elapsed >= last_vis_time + config.val_check_interval and config.visualize_progress:
        #     # 评估当前模型
        #     test_accuracy = evaluate_model(models[begin_idx], criterion, test_loader, device)
            
        #     elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
        #     print(f"\nStep {begin_idx}/{T} | Test Acc: {test_accuracy:.2f}% | Time: {elapsed_str}")
            
        #     last_vis_time = elapsed
    
    # 训练结束
    pbar.close()
    

    
    # 最终测试
    final_model = models[T]
    final_accuracy = evaluate_model(final_model, criterion, test_loader, device)
    
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Total iterations: {total_iters} (vs {T} normal iterations)")
    print(f"Effective speed-up: {T/total_iters:.2f}x")
    
    return final_model

def train_loop_serial(config, model, criterion, train_loader, test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    
    # 训练循环
    start_time = time.time()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    pbar = tqdm(total=config.max_steps)
    
    data_iter = iter(train_loader)
    for step in range(config.max_steps):
        # 获取数据
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        
        images, labels = images.to(device), labels.to(device)
        
        # 前向和后向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
        
        # 更新统计数据
        running_loss += loss.item()
        running_correct += correct
        running_total += labels.size(0)
        
        # 更新进度条
        if (step + 1) % 5 == 0:
            avg_loss = running_loss / (step + 1)
            avg_acc = 100 * running_correct / running_total
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            
            pbar.set_description(
                f'Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | Time: {elapsed_str}'
            )
        
        pbar.update(1)
    
    pbar.close()
    
    # 最终测试
    test_accuracy = evaluate_model(model, criterion, test_loader, device)
    
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    return model

def evaluate_model(model, criterion, test_loader, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    model.train()
    return accuracy

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='ParaOpt_mul_v1')

    parser.add_argument('--device',type=str,help='Choose CUDA device')

    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--max_steps', type=int, help='max steps')
    parser.add_argument('--learning_rate', type=float, help='learning rate')

    parser.add_argument('--P', type=int, help='window size')
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--ema_decay', type=float, help='阈值指数移动平均衰减率')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='the computation type of error_all')

    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='the type of optimizer')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='simulation_parallel or serial')
    
    # 新增的模型选择参数
    parser.add_argument('--model_name', type=str, 
                        choices=['cnn', 'resnet18', 'resnet34', 'resnet50', 
                                 'vgg16', 'vgg19', 'densenet121', 
                                 'mobilenet_v2', 'efficientnet_b0'], 
                        help='neural network model to use')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    
    return parser

def main():

    parser = setup_arg_parser()
    args = parser.parse_args()

    config = Config()
    config.update_from_args(args)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=config.batch_size, shuffle=True)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 创建模型和损失函数
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn',force=True)
    queues = mp.Queue(),mp.Queue(),mp.Queue()

    processes = []
    num_processes = config.P + 1


    if num_processes == 1:
        run(0,1,queues,config,model,criterion,train_loader,test_loader)
        exit(0)

    for rank in range(num_processes):
        p = mp.Process(target=run,args=(rank,num_processes,queues,config,model,criterion,train_loader,test_loader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    # 保存最终模型
    # torch.save(trained_model.state_dict(), 'cnn_model.pth')
    # print("Model saved to 'cnn_model.pth'")

if __name__ == "__main__":
    main()