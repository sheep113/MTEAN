"""
DNA Whisper模型训练入口脚本 - 支持配置驱动的训练流程
"""
import os
import sys
import json
import logging
import argparse
import time
import glob
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
# 移除 tensorboard 导入

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from utils.hdf5_dataset import create_data_loaders, HDF5Dataset
from training.models.model import DNAWhisper
# 从train模块导入DNAWhisperTrainer
from training.train import DNAWhisperTrainer, Trainer

def setup_logging(log_dir: Path) -> logging.Logger:
    """设置日志记录"""
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logger = logging.getLogger("DNAWhisperTrainer")
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_fmt)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_fmt)
    
    # 添加处理器
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def set_seed(seed: int) -> None:
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# 添加简单的训练日志记录器，替代 TensorBoard
class SimpleLogger:
    """简单的训练日志记录器，替代TensorBoard"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_file = self.log_dir / "metrics.csv"
        
        # 初始化metrics文件
        with open(self.metrics_file, 'w') as f:
            f.write("step,metric,value\n")
    
    def add_scalar(self, tag, value, global_step):
        """记录标量值到CSV文件"""
        with open(self.metrics_file, 'a') as f:
            f.write(f"{global_step},{tag},{value}\n")
    
    def close(self):
        """关闭日志记录器（这里不需要操作）"""
        pass

def find_hdf5_files(directory_path):
    """在目录中查找HDF5文件"""
    h5_files = list(glob.glob(os.path.join(directory_path, "*.h5")))
    hdf5_files = list(glob.glob(os.path.join(directory_path, "*.hdf5")))
    all_files = h5_files + hdf5_files
    
    if not all_files:
        # 如果当前目录没有找到HDF5文件，尝试在子目录中查找
        for subdir in os.listdir(directory_path):
            subdir_path = os.path.join(directory_path, subdir)
            if os.path.isdir(subdir_path):
                subdir_files = find_hdf5_files(subdir_path)
                all_files.extend(subdir_files)
    
    return all_files

def main(args):
    """主程序入口"""
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 设置日志记录
    logger = setup_logging(output_dir)
    logger.info(f"开始训练 - 输出目录: {output_dir}")
    
    # 加载配置文件
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 保存配置副本到输出目录
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # 设置随机种子
    seed = config.get("random_seed", 42)
    set_seed(seed)
    logger.info(f"设置随机种子: {seed}")
    
    # 设置设备，分布式训练
    local_rank = args.local_rank
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    
    device = torch.device(f'cuda:{local_rank}' if local_rank != -1 else 'cuda' if torch.cuda.is_available() else 'cpu')
    is_main_process = local_rank in [-1, 0]
    
    if is_main_process:
        logger.info(f"使用设备: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # 使用简单日志记录器替代TensorBoard
    metrics_logger = None
    if is_main_process:
        metrics_logger = SimpleLogger(log_dir=str(output_dir / "metrics"))
    
    # 加载数据集
    dataset_path = Path(args.data_path)
    if not dataset_path.exists():
        logger.error(f"数据集不存在: {dataset_path}")
        return 1
    
    # 加载分区文件
    region_partition_file = dataset_path / "region_partition.npy"
    if not region_partition_file.exists():
        logger.error(f"分区文件不存在: {region_partition_file}")
        return 1
        
    region_partition = np.load(str(region_partition_file))
    logger.info(f"加载分区文件: {region_partition_file}, 形状: {region_partition.shape}")
    
    # 处理数据集路径
    if dataset_path.is_dir():
        # 如果提供的是目录，查找目录下的HDF5文件
        hdf5_files = find_hdf5_files(str(dataset_path))
        if not hdf5_files:
            logger.error(f"在目录 {dataset_path} 或其子目录中未找到HDF5文件")
            return 1
        
        logger.info(f"在目录 {dataset_path} 中找到 {len(hdf5_files)} 个HDF5文件")
        # 使用第一个HDF5文件作为主数据集
        dataset_path = Path(hdf5_files[0])
        logger.info(f"使用数据集文件: {dataset_path}")
    
    # 获取数据集元数据
    metadata = HDF5Dataset.get_metadata(str(dataset_path))
    feature_dim = metadata.get('data_shape', (0, 0, 7))[-1]
    
    if is_main_process:
        # 提取序列长度（通常是数据形状的第二维）
        seq_length = metadata.get('data_shape', (0, 0, 0))[1] if len(metadata.get('data_shape', [])) > 2 else 0
        logger.info(f"数据集特征维度: {feature_dim}")
        logger.info(f"数据集形状: {metadata.get('data_shape')}")
        logger.info(f"输入序列长度: {seq_length}")
        
        # 从配置中获取分块数（如果有的话）
        chunk_size = config.get("model", {}).get("chunk_size", 0)
        num_chunks = seq_length // chunk_size if chunk_size > 0 else 1
        logger.info(f"分块大小: {chunk_size}, 分块数: {num_chunks}")
        
    # 使用DNAWhisperTrainer进行训练
    # 注意: 我们将使用高级训练器来简化训练流程
    try:
        trainer = DNAWhisperTrainer(
            config_path=str(config_path), 
            output_dir=str(output_dir),
            region_partition=region_partition  
        )
        # 设置数据集和特征维度
        trainer.logger = logger
        trainer.feature_dim = feature_dim
        
        # 创建数据加载器
        batch_size = config["training"]["batch_size"]
        num_workers = min(4, os.cpu_count() or 1)
        
        # 设置表型维度，从配置或元数据中获取，如果都没有则设为默认值
        phenotype_dim = config.get('output_layer', {}).get('phenotype_dim', 1)
        trainer.phenotype_dim = phenotype_dim
        if is_main_process:
            logger.info(f"表型维度设置为: {phenotype_dim}")

        data_loaders = create_data_loaders(
            str(dataset_path),
            batch_size=batch_size,
            num_workers=num_workers,
            distributed=(local_rank != -1),
            local_rank=local_rank,
            world_size=world_size
        )
        
        # 将数据加载器传递给训练器
        trainer.train_dataset = data_loaders['train'].dataset
        trainer.val_dataset = data_loaders['valid'].dataset
        trainer.test_dataset = data_loaders['test'].dataset if 'test' in data_loaders else None
        
        # 设置接收分布式训练参数
        trainer.local_rank = local_rank
        trainer.is_distributed = local_rank != -1
        
        # 设置日志记录器（替代TensorBoard）
        trainer.tb_writer = metrics_logger
        
        # 初始化模型
        trainer.setup_model()
        
        # 验证分区机制
        if is_main_process:
            logger.info("="*50)
            logger.info("验证模型分区机制...")
            model = trainer.trainer.model
            if hasattr(model, '_identify_regions'):
                # 获取一小批数据作为测试
                test_batch = next(iter(data_loaders['train']))
                test_snps = test_batch[0].to(device)
                batch_size, seq_len = test_snps.shape[0], test_snps.shape[1]
                
                # 从特征中提取区域标识符(索引为6)
                has_region_feature = test_snps.shape[-1] >= 7
                if has_region_feature:
                    region_markers = torch.round(test_snps[:, :, 6]).long()
                    
                    # 测试区域识别功能
                    region_boundaries = model._identify_regions(region_markers)
                    
                    # 检查边界的有效性
                    valid_boundaries = True
                    overlap_detected = False
                    missing_regions = False
                    
                    # 输出分区统计信息
                    total_regions = sum(len(boundaries) for boundaries in region_boundaries)
                    regions_per_sample = [len(boundaries) for boundaries in region_boundaries]
                    avg_regions = total_regions / batch_size
                    
                    # 检查每个样本的区域边界
                    for i, boundaries in enumerate(region_boundaries):
                        if not boundaries:  # 空边界
                            missing_regions = True
                            logger.warning(f"样本 {i} 未检测到任何区域边界")
                            continue
                            
                        # 检查边界的有序性和完整性
                        prev_end = 0
                        for j, (start, end) in enumerate(boundaries):
                            # 检查边界有效性
                            if start >= end:
                                valid_boundaries = False
                                logger.warning(f"样本 {i} 的区域 {j} 边界无效: [{start}, {end})")
                                
                            # 检查是否有间隙
                            if start > prev_end:
                                logger.warning(f"样本 {i} 在位置 {prev_end}-{start} 之间存在未覆盖的序列")
                                
                            # 检查是否有重叠
                            if start < prev_end:
                                overlap_detected = True
                                logger.warning(f"样本 {i} 的区域 {j} 与前一区域重叠: 前区域结束于 {prev_end}, 当前区域开始于 {start}")
                                
                            prev_end = end
                            
                        # 检查是否覆盖了整个序列
                        if boundaries[-1][1] < seq_len:
                            logger.warning(f"样本 {i} 的最后一个区域未覆盖整个序列: 区域结束于 {boundaries[-1][1]}, 序列长度为 {seq_len}")
                    
                    logger.info(f"分区验证结果:")
                    logger.info(f"- 特征维度: {test_snps.shape[-1]}")
                    logger.info(f"- 序列长度: {seq_len}")
                    logger.info(f"- 批次大小: {batch_size}")
                    logger.info(f"- 识别的总区域数: {total_regions}")
                    logger.info(f"- 平均每个样本的区域数: {avg_regions:.2f}")
                    logger.info(f"- 每个样本的区域数分布: {min(regions_per_sample)}-{max(regions_per_sample)}")
                    logger.info(f"- 区域边界有效: {'是' if valid_boundaries else '否'}")
                    logger.info(f"- 检测到区域重叠: {'是' if overlap_detected else '否'}")
                    logger.info(f"- 存在缺失区域: {'是' if missing_regions else '否'}")
                    
                    # 显示第一个样本的详细区域信息
                    if len(region_boundaries) > 0 and len(region_boundaries[0]) > 0:
                        logger.info(f"- 第一个样本的区域边界: {region_boundaries[0]}")
                        # 统计第一个样本的区域长度分布
                        region_lens = [end-start for start, end in region_boundaries[0]]
                        logger.info(f"- 第一个样本的区域长度: 最小={min(region_lens)}, 最大={max(region_lens)}, 平均={sum(region_lens)/len(region_lens):.2f}")
                        
                    logger.info("分区机制验证完成")
                else:
                    logger.warning("输入特征维度不足7，无法进行区域分区!")
            else:
                logger.warning("模型中未找到_identify_regions方法，分区机制可能未实现!")
            logger.info("="*50)
        
        # 如果指定了继续训练，加载检查点
        if args.resume:
            if is_main_process:
                logger.info(f"从检查点恢复训练: {args.resume}")
            trainer.resume_from = args.resume
        
        # 开始训练
        logger.info("开始训练过程...")
        train_results = trainer.train(resume_from=args.resume if args.resume else None)
        
        if is_main_process:
            # 记录训练结果到文件
            if hasattr(train_results, 'history') and train_results.history:
                with open(output_dir / "training_history.json", 'w') as f:
                    json.dump(train_results.history, f, indent=2)
                
                # 绘制训练曲线
                if 'train_loss' in train_results.history and 'val_loss' in train_results.history:
                    plot_training_curves(
                        train_results.history['train_loss'],
                        train_results.history['val_loss'],
                        output_dir / "training_curves.png"
                    )
            
            # 评估最佳模型
            logger.info("开始测试...")
            best_model_path = os.path.join(output_dir, "checkpoints", "best_model.pt")
            if os.path.exists(best_model_path):
                trainer.trainer.model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
                test_loss = trainer.trainer.validate(data_loaders['test'])
                logger.info(f"测试完成 - 测试损失: {test_loss:.6f}")
            else:
                logger.warning(f"最佳模型不存在: {best_model_path}")
        
        # 关闭日志记录器
        if is_main_process and metrics_logger is not None:
            metrics_logger.close()
            
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info("训练完成!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练DNA Whisper模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--data-path", type=str, required=True, help="HDF5数据集路径")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")
    
    args = parser.parse_args()
    sys.exit(main(args))
