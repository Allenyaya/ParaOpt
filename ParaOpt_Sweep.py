import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
import argparse
import copy
import os
import torchvision.models as models
import wandb

class Config:

    device = 2

    # 训练参数
    batch_size = 32
    num_epochs = 1
    max_steps = 10000  # 总迭代步数
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

    optimizer_type = 'SGD' # sgd,adam,adamw

    # 训练模式
    training_mode = 'parallel'
    
    # 模型配置
    model_name = 'cnn'  # 默认使用自定义CNN
    num_classes = 10    # 分类数量
    pretrained = False  # 是否使用预训练模型

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
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
        return error / total_num * 1e6  # 缩放以便于比较

def train_with_config(config_dict=None):
    """专为wandb.sweep设计的训练函数"""
    
    # 初始化基本配置
    config = Config()
    
    # 如果wandb传入了配置参数，则更新
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # 初始化wandb
    run = wandb.init(project='NIPS_2025_ParaOptimizer', config=config_dict)
    
    # 使用wandb的config更新参数（允许sweep覆盖）
    for key, value in wandb.config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # 数据预处理和加载  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 使用模型工厂创建模型
    model = ModelFactory.create_model(config)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Using model: {config.model_name}")
    
    if config.training_mode.lower() == 'parallel':
        trained_model = train_loop_parallel_simulation(config, model, criterion, train_loader, test_loader)
    else:
        trained_model = train_loop_serial(config, model, criterion, train_loader, test_loader)
    
    # 评估最终测试准确率
    final_accuracy = evaluate_model(trained_model, criterion, test_loader, 
                                   torch.device(config.device if torch.cuda.is_available() else "cpu"))
    
    # 记录最终指标
    wandb.log({"final_test_accuracy": final_accuracy})
    
    # 可选：保存最终模型
    model_save_path = f'{config.model_name}_model.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'")
    
    # 关闭wandb run
    wandb.finish()
    
    return final_accuracy


def setup_sweep_configuration():
    """设置wandb sweep配置"""
    sweep_config = {
        'method': 'random',  
        'metric': {
            'name': 'final_test_accuracy',  
            'goal': 'maximize' 
        },
        'parameters': {
            'learning_rate': {
                'distribution':'log_uniform_values',
                'min': 1e-6,
                'max': 0.1                
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
                # 'distribution':'q_uniform',
                # 'q': 8,
                # 'min': 16,
                # 'max': 256,
            },
            'optimizer_type': {
                'values': ['SGD', 'Adam', 'AdamW']
            },
            'P': {  
                # 'values': [5, 7, 10, 15]
                'distribution':'q_uniform',
                'q': 1,
                'min': 2,
                'max': 15,                
            },
            'threshold': {              
                'min': 1e-6,
                'max': 1e-4
            },
            'ema_decay': {
                'min': 0.8,
                'max': 0.99
            },
            'adaptivity_type': {
                'values': ['mean', 'median']
            },
            # 'model_name': {
            #     'values': ['cnn', 'resnet18', 'mobilenet_v2']
            # },
            # 'training_mode': {
            #     'values': ['parallel', 'serial']
            # }
        }
    }
    
    return sweep_config

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)
        
    def clone(self, device, other=None):
        if other is None:
            other = ModelWrapper(copy.deepcopy(self.model)).to(device)
        else:
            with torch.no_grad():
                other_params = other.get_params()
                my_params = self.get_params()
                for param_to, param_from in zip(other_params, my_params):
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
        return error / total_num * 1e6  # 缩放以便于比较


class ModelFactory:
    @staticmethod
    def create_model(config):
        model_name = config.model_name.lower()
        num_classes = config.num_classes
        pretrained = config.pretrained

        # 自定义CNN模型
        if model_name == 'cnn':
            return CNN(num_classes=num_classes)
            
        # ResNet系列
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return ModelWrapper(model)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return ModelWrapper(model)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return ModelWrapper(model)
            
        # VGG系列
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return ModelWrapper(model)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return ModelWrapper(model)
            
        # DenseNet系列
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return ModelWrapper(model)
            
        # MobileNet系列
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return ModelWrapper(model)
            
        # EfficientNet系列
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return ModelWrapper(model)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())


def take_step(model, optimizer, criterion, dataloader, data_iter, step, seed_offset):
    # 设置随机种子以确保可重现性
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    # 获取批次数据
    try:
        images, labels = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
    
    # 将数据移至GPU
    device = next(model.parameters()).device
    images, labels = images.to(device), labels.to(device)
    
    # 前向和后向传播
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # 保存梯度
    grads = [param.grad.clone() for param in model.parameters()]
    
    # 计算准确率
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / labels.size(0)
    
    return {
        'grads': grads,
        'loss': loss.item(),
        'accuracy': accuracy,
        'data_iter': data_iter,
        'batch_size': labels.size(0),
        'correct': correct
    }


def train_loop_parallel_simulation(config, model, criterion, train_loader, test_loader):
    #在单GPU上使用串行模拟并行处理的训练循环

    device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    T = config.max_steps
    P = min(config.P, T//2)
    thresh = config.threshold
    
    
    models = [None for _ in range(T+1)]
    optimizers = [None for _ in range(T+1)]
    

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
        if config.optimizer_type.lower()=='sgd':
            optimizers[step] = optim.SGD(models[step].parameters(), 
                                        lr=config.learning_rate, 
                                        momentum=config.momentum)
        elif config.optimizer_type.lower()=='adam':
            optimizers[step] = optim.Adam(models[step].parameters(),
                                        lr=config.learning_rate,
                                        betas=(0.9,0.999),
                                        eps=1e-08)
        elif config.optimizer_type.lower()=='adamw':
            optimizers[step] = optim.AdamW(models[step].parameters(),
                                         lr=config.learning_rate,
                                         betas=(0.9,0.999),
                                         eps=1e-08)

    # 主训练循环
    data_iter = iter(train_loader)
    while begin_idx < T:
        # 计算窗口大小
        parallel_len = end_idx - begin_idx
        
        # 存储梯度预测和指标
        pred_f = [None for _ in range(parallel_len)]
        metrics = [None for _ in range(parallel_len)]
        
        # 串行模拟并行计算梯度
        for i in range(parallel_len):
            step = begin_idx + i
            
            # 串行计算每个模型的梯度
            result = take_step(models[step], optimizers[step], criterion, 
                              train_loader, data_iter, step, config.seed)
            
            # 更新数据迭代器
            data_iter = result['data_iter']
            
            # 存储梯度和指标
            pred_f[i] = result['grads']
            metrics[i] = {
                'loss': result['loss'],
                'accuracy': result['accuracy'],
                'batch_size': result['batch_size'],
                'correct': result['correct']
            }
            
            # 更新统计数据
            running_loss += result['loss']
            running_correct += result['correct']
            running_total += result['batch_size']
        
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
            wandb.log({"Train_Acc":accuracy,"Train_Loss":avg_loss,"Iter":total_iters,"Steps":begin_idx})
        
        test_accuracy = evaluate_model(models[begin_idx], criterion, test_loader, device)
        wandb.log({"Test_Acc":test_accuracy,"Iter":total_iters,"Steps":begin_idx})

        # 定期验证
        elapsed = time.time() - start_time
        if elapsed >= last_vis_time + config.val_check_interval and config.visualize_progress:
            # 评估当前模型
            # test_accuracy = evaluate_model(models[begin_idx], criterion, test_loader, device)
            
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            print(f"\nStep {begin_idx}/{T} | Test Acc: {test_accuracy:.2f}% | Time: {elapsed_str}")
            
            # wandb.log({"Test_Acc":test_accuracy,"Step":begin_idx})
            # wandb.log({"Test_Acc":test_accuracy,"Iter":total_iters,"Steps":begin_idx})
            last_vis_time = elapsed
    
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


def evaluate_model(model, criterion, test_loader, device):
    # 评估模型性能
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


def train_loop_serial(config, model, criterion, train_loader, test_loader):
    # 标准的串行训练循环
    device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    total_iters = 0
    
    if config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif config.optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
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
        
        # 更新统计数据
        running_loss += loss.item()
        running_correct += correct
        running_total += labels.size(0)
        
        train_acc = 100 * running_correct / running_total
        avg_loss =  running_loss / (step + 1)
        test_acc = evaluate_model(model, criterion, test_loader, device)
        
        total_iters += 1
        
        wandb.log({"Train_Acc":train_acc,"Train_Loss":avg_loss,"Test_Acc":test_acc,"Iter":total_iters})

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


def setup_arg_parser():
    parser = argparse.ArgumentParser(description='ParaSGD_simul with wandb sweep')

    parser.add_argument('--sweep', action='store_true', help='Run wandb sweep')
    parser.add_argument('--agent', action='store_true', help='Run as a sweep agent')
    parser.add_argument('--sweep_id', type=str, help='Sweep ID to use for agent')
    parser.add_argument('--count', type=int, default=10, help='Number of runs for the sweep')

    parser.add_argument('--device', type=int, help='Choose CUDA device')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--max_steps', type=int, help='max steps')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--P', type=int, help='window size')
    parser.add_argument('--threshold', type=float, help='threshold')
    parser.add_argument('--ema_decay', type=float, help='阈值指数移动平均衰减率')
    parser.add_argument('--adaptivity_type', type=str, choices=['mean', 'median'], help='the computation type of error_all')
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adam', 'adamw'], help='the type of optimizer')
    parser.add_argument('--training_mode', type=str, choices=['parallel', 'serial'], help='simulation_parallel or serial')
    parser.add_argument('--model_name', type=str, 
                        choices=['cnn', 'resnet18', 'resnet34', 'resnet50', 
                                 'vgg16', 'vgg19', 'densenet121', 
                                 'mobilenet_v2', 'efficientnet_b0'], 
                        help='neural network model to use')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')  
    
    
    return parser


def main():
    parser =setup_arg_parser()

    args = parser.parse_args()
    
    if args.sweep:
        # 创建新的sweep
        sweep_config = setup_sweep_configuration()
        sweep_id = wandb.sweep(sweep_config, project='NIPS_2025_ParaOptimizer')
        print(f"Created sweep with ID: {sweep_id}")
        
        if args.agent:
            # 直接运行代理
            wandb.agent(sweep_id, function=train_with_config, count=args.count)
    
    elif args.agent and args.sweep_id:
        # 使用现有的sweep ID运行代理
        wandb.agent(args.sweep_id, function=train_with_config, project='NIPS_2025_ParaOptimizer', count=args.count)
    
    else:
        # 常规的单次运行，使用命令行参数
        config = Config()
        config.update_from_args(args)
        
        # 创建从args转换来的config_dict
        config_dict = {k: v for k, v in vars(args).items() if hasattr(config, k) and v is not None}
        train_with_config(config_dict)


if __name__ == "__main__":
    main()