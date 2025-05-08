import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
import copy
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset

# 配置类定义
class LLMConfig:
    # 训练参数
    batch_size = 4
    num_epochs = 1
    max_steps = 10000  # 总迭代步数
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_steps = 500
    
    # 模型配置
    model_name = "gpt2"  # 可以是 "gpt2", "facebook/opt-125m", "bert-base-uncased" 等
    max_length = 512  # 最大序列长度
    
    # 并行配置
    P = 4  # 窗口大小
    threshold = 1e-5  # 误差阈值
    ema_decay = 0.9  # 阈值指数移动平均衰减率
    adaptivity_type = 'mean'  # 自适应策略: 'mean' 或 'median'
    val_check_interval = 60  # 验证间隔(秒)
    
    # 系统配置
    seed = 42  # 随机种子
    device_count = torch.cuda.device_count()  # GPU数量
    gradient_accumulation_steps = 4  # 梯度累积步数

# 数据预处理
class TextDataLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载数据集 (这里使用wikitext作为示例)
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        
        # 预处理函数
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        # 处理数据集
        self.train_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        self.eval_dataset = dataset["validation"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        self.eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    def get_train_loader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
    
    def get_eval_loader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

# 设备转移函数
def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

# 大语言模型包装器
class LLMWrapper(nn.Module):
    def __init__(self, config):
        super(LLMWrapper, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.config = config
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def clone(self, device, other=None):
        if other is None:
            other = LLMWrapper(self.config).to(device)
            
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

# 优化器状态克隆
def optimizer_state_clone(optimizer_from, optimizer_to):
    optimizer_to.load_state_dict(optimizer_from.state_dict())

# 训练步骤函数
def take_step(model, optimizer, scheduler, criterion, dataloader, data_iter, device, step, seed_offset):
    """执行单个训练步骤并返回梯度"""
    # 设置随机种子以确保可重现性
    np.random.seed(step + seed_offset)
    torch.manual_seed(step + seed_offset)
    torch.cuda.manual_seed(step + seed_offset)
    
    # 获取批次数据
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)
    
    batch = to_device(batch, device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = input_ids.clone()
    
    # 前向和后向传播
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    # 计算准确率
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = torch.numel(labels)
    
    return {
        'loss': loss.item(),
        'accuracy': 100 * correct / total,
        'data_iter': data_iter,
        'batch_size': input_ids.size(0),
        'correct': correct,
        'total': total
    }

# 工作进程函数
def run_worker(model, optimizer, scheduler, criterion, dataloader, queues, device, seed_offset):
    """工作进程函数 - 执行梯度计算"""
    data_iter = iter(dataloader)
    
    while True:
        ret = queues[0].get()
        if ret is None:
            return
            
        params, step = ret
        model.set_params(params, clone=False)
        
        res = take_step(model, optimizer, scheduler, criterion, dataloader, data_iter, device, step, seed_offset)
        data_iter = res['data_iter']  # 更新数据迭代器
        
        # 收集计算的梯度
        my_params = model.get_params()
        grads = [param.grad for param in my_params]
        
        # 将梯度和指标发送回主进程
        queues[1].put((grads, step, {
            'loss': res['loss'], 
            'accuracy': res['accuracy'], 
            'batch_size': res['batch_size'], 
            'correct': res['correct'],
            'total': res['total']
        }))

# 主训练循环
def train_loop_parallel(config, model, criterion, train_loader, eval_loader):
    """使用定点迭代和滑动窗口的并行训练循环"""
    # 如果只有一个GPU，退回到常规训练
    if config.device_count <= 1:
        return train_loop_serial(config, model, criterion, train_loader, eval_loader)
        
    # 初始化多进程
    mp.set_start_method('spawn', force=True)
    queues = (mp.Queue(), mp.Queue())
    
    # 创建工作进程
    processes = []
    worker_devices = []
    
    for rank in range(1, config.device_count):
        worker_device = torch.device(f"cuda:{rank}")
        worker_devices.append(worker_device)
        worker_model = LLMWrapper(config).to(worker_device)
        
        # 为每个工作进程创建优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in worker_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in worker_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        worker_optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        worker_scheduler = get_linear_schedule_with_warmup(
            worker_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
                                    
        p = mp.Process(
            target=run_worker, 
            args=(worker_model, worker_optimizer, worker_scheduler, criterion, 
                  train_loader, queues, worker_device, config.seed)
        )
        p.start()
        processes.append(p)
    
    # 设置主进程参数
    device = torch.device("cuda:0")
    model = model.to(device)
    
    # 为主进程创建优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    T = config.max_steps
    P = min(config.P, T)  # 调整窗口大小不超过GPU数量
    thresh = config.threshold
    
    # 初始化模型和优化器数组
    models = [None for _ in range(T+1)]
    optimizers = [None for _ in range(T+1)]
    schedulers = [None for _ in range(T+1)]
    
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
        
        # 为每个步骤创建优化器
        optimizers[step] = optim.AdamW(
            optimizer_grouped_parameters, 
            lr=config.learning_rate
        )
        schedulers[step] = get_linear_schedule_with_warmup(
            optimizers[step],
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
    
    # 主训练循环
    data_iter = iter(train_loader)
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
            running_total += _metrics['total']
        
        # 从窗口起点开始执行定点迭代
        rollout_model = models[begin_idx]
        rollout_optimizer = optimizers[begin_idx]
        rollout_scheduler = schedulers[begin_idx]
        
        ind = None  # 重新同步点
        errors_all = 0  # 累积误差
        
        # 对窗口中的每个位置执行定点迭代
        for i in range(parallel_len):
            step = begin_idx + i
            
            # 设置之前计算的梯度并执行优化步骤
            rollout_model.set_grads_from_grads(pred_f[i])
            rollout_optimizer.step()
            rollout_scheduler.step()
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
                optimizer_state_clone(rollout_scheduler, schedulers[step+1])
            
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
            models[step] = rollout_model.clone(
                device, 
                models[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None
            )
            optimizers[step] = optimizers[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None
            schedulers[step] = schedulers[step - 1 - parallel_len] if step - 1 - parallel_len >= 0 else None
        
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
        elapsed = time.time() - start_time
        if elapsed >= last_vis_time + config.val_check_interval:
            # 评估当前模型
            eval_accuracy, eval_loss = evaluate_model(models[begin_idx], criterion, eval_loader, device)
            
            elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
            print(f"\nStep {begin_idx}/{T} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_accuracy:.2f}% | Time: {elapsed_str}")
            
            last_vis_time = elapsed
    
    # 训练结束
    pbar.close()
    
    # 关闭工作进程
    for _ in range(len(processes)):
        queues[0].put(None)
    for p in processes:
        p.join()
    
    # 最终评估
    final_model = models[T]
    eval_accuracy, eval_loss = evaluate_model(final_model, criterion, eval_loader, device)
    
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final eval loss: {eval_loss:.4f} | Final eval accuracy: {eval_accuracy:.2f}%")
    print(f"Total iterations: {total_iters} (vs {T} normal iterations)")
    print(f"Effective speed-up: {T/total_iters:.2f}x")
    
    return final_model

# 串行训练循环
def train_loop_serial(config, model, criterion, train_loader, eval_loader):
    """常规的串行训练循环"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
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
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        
        # 前向和后向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # 梯度累积
        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 计算准确率
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == labels).sum().item()
        total = torch.numel(labels)
        
        # 更新统计数据
        running_loss += loss.item()
        running_correct += correct
        running_total += total
        
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
    
    # 最终评估
    eval_accuracy, eval_loss = evaluate_model(model, criterion, eval_loader, device)
    
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed))).split('.')[0]
    print(f"\nTraining completed in {elapsed_str}")
    print(f"Final eval loss: {eval_loss:.4f} | Final eval accuracy: {eval_accuracy:.2f}%")
    
    return model

# 评估函数
def evaluate_model(model, criterion, eval_loader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = to_device(batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += torch.numel(labels)
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(eval_loader)
    model.train()
    return accuracy, avg_loss

# 主函数
def main():
    # 加载配置
    config = LLMConfig()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 加载数据和模型
    data_loader = TextDataLoader(config)
    train_loader = data_loader.get_train_loader()
    eval_loader = data_loader.get_eval_loader()
    
    model = LLMWrapper(config)
    criterion = nn.CrossEntropyLoss()
    
    # 使用并行训练循环
    trained_model = train_loop_parallel(config, model, criterion, train_loader, eval_loader)
    
    # 保存最终模型
    output_dir = "./saved_model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trained_model.model.save_pretrained(output_dir)
    data_loader.tokenizer.save_pretrained(output_dir)
    print(f"\nModel and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()