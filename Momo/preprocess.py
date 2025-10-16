"""
NinaproDB2 数据预处理脚本
用于EMG和IMU多模态手势识别任务
"""

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import h5py
import pickle
from pathlib import Path


class NinaproDB2Preprocessor:
    """NinaproDB2数据集预处理器"""
    
    def __init__(self, data_root, subjects=[10, 23, 36], fs=2000):
        """
        初始化预处理器
        
        参数:
            data_root: 数据根目录
            subjects: 受试者列表
            fs: 采样率 (Hz)
        """
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.fs = fs
        
        # 窗口参数
        self.window_size = 200  # ms
        self.window_step = 25   # ms
        self.window_samples = int(self.window_size * self.fs / 1000)  # 400 samples
        self.step_samples = int(self.window_step * self.fs / 1000)    # 50 samples
        
        # 排除过渡期（运动边界前后100ms）
        self.transition_margin = int(0.1 * self.fs)  # 200 samples
        
        # 数据分割（重复索引）
        self.train_reps = [1, 2, 3, 4, 6]  # 训练集（重复1, 2, 3, 4, 6）
        self.test_reps = [2, 5]             # 测试集（重复2, 5）
        
        print(f"初始化预处理器 - 采样率: {self.fs} Hz")
        print(f"窗口大小: {self.window_size} ms ({self.window_samples} samples)")
        print(f"窗口步长: {self.window_step} ms ({self.step_samples} samples)")
        
    def load_subject_data(self, subject):
        """
        加载单个受试者的所有练习数据
        
        参数:
            subject: 受试者编号
            
        返回:
            emg_data: EMG数据列表
            imu_data: IMU数据列表
            labels: 标签列表
            repetitions: 重复索引列表
        """
        print(f"\n加载受试者 S{subject} 数据...")
        
        subject_dir = self.data_root / f"DB2_s{subject}"
        emg_list, imu_list, label_list, rep_list = [], [], [], []
        
        # 加载3个练习（E1, E2, E3）
        for exercise in [1, 2, 3]:
            mat_file = subject_dir / f"S{subject}_E{exercise}_A1.mat"
            
            if not mat_file.exists():
                print(f"  警告: 文件不存在 {mat_file}")
                continue
            
            print(f"  加载练习 E{exercise}: {mat_file.name}")
            
            # 加载.mat文件
            mat_data = sio.loadmat(str(mat_file))
            
            # 提取数据
            emg = mat_data['emg']  # (N, 12)
            acc = mat_data['acc']  # (N, 36)
            restimulus = mat_data['restimulus'].flatten()  # (N,)
            rerepetition = mat_data['rerepetition'].flatten()  # (N,)
            
            # 检查数据形状
            print(f"    EMG形状: {emg.shape}, IMU形状: {acc.shape}")
            print(f"    标签范围: {np.min(restimulus)}-{np.max(restimulus)}")
            print(f"    重复范围: {np.min(rerepetition)}-{np.max(rerepetition)}")
            
            # 检查NaN值
            if np.any(np.isnan(emg)) or np.any(np.isnan(acc)):
                print(f"    警告: 检测到NaN值，使用线性插值填充")
                emg = self._interpolate_nan(emg)
                acc = self._interpolate_nan(acc)
            
            emg_list.append(emg)
            imu_list.append(acc)
            label_list.append(restimulus)
            rep_list.append(rerepetition)
        
        return emg_list, imu_list, label_list, rep_list
    
    def _interpolate_nan(self, data):
        """对NaN值进行线性插值"""
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            nans = np.isnan(channel_data)
            if np.any(nans):
                x = np.arange(len(channel_data))
                channel_data[nans] = np.interp(
                    x[nans], x[~nans], channel_data[~nans]
                )
                data[:, ch] = channel_data
        return data
    
    def bandpass_filter(self, data, lowcut=10, highcut=500, order=4):
        """
        应用Butterworth带通滤波器
        
        参数:
            data: 输入数据 (N, C)
            lowcut: 低频截止 (Hz)
            highcut: 高频截止 (Hz)
            order: 滤波器阶数
            
        返回:
            filtered_data: 滤波后的数据
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        
        # 对每个通道应用零相移滤波
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
        
        return filtered_data
    
    def highpass_filter(self, data, cutoff=0.5, order=4):
        """
        应用高通滤波器（用于IMU去趋势）
        
        参数:
            data: 输入数据 (N, C)
            cutoff: 截止频率 (Hz)
            order: 滤波器阶数
            
        返回:
            filtered_data: 滤波后的数据
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
        
        return filtered_data
    
    def rectify_emg(self, emg):
        """EMG全波整流"""
        return np.abs(emg)
    
    def smooth_emg_rms(self, emg, window_ms=200, step_ms=50):
        """
        使用RMS窗口计算EMG包络
        
        参数:
            emg: EMG数据 (N, C)
            window_ms: RMS窗口大小 (ms)
            step_ms: RMS步长 (ms)
            
        返回:
            smoothed: 平滑后的EMG
        """
        window_samples = int(window_ms * self.fs / 1000)
        step_samples = int(step_ms * self.fs / 1000)
        
        n_samples = emg.shape[0]
        n_channels = emg.shape[1]
        n_windows = (n_samples - window_samples) // step_samples + 1
        
        smoothed = np.zeros((n_windows, n_channels))
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            window_data = emg[start:end, :]
            # 计算RMS
            smoothed[i, :] = np.sqrt(np.mean(window_data**2, axis=0))
        
        # 上采样回原始长度
        if n_windows > 1:
            x_old = np.linspace(0, n_samples-1, n_windows)
            x_new = np.arange(n_samples)
            smoothed_full = np.zeros((n_samples, n_channels))
            for ch in range(n_channels):
                f = interp1d(x_old, smoothed[:, ch], kind='linear', 
                            fill_value='extrapolate')
                smoothed_full[:, ch] = f(x_new)
            return smoothed_full
        else:
            return emg
    
    def preprocess_emg(self, emg):
        """
        EMG预处理流程
        
        步骤：
        1. 带通滤波 (10-500 Hz)
        2. 全波整流
        3. RMS平滑
        """
        print("  处理EMG: 带通滤波 -> 整流 -> RMS平滑")
        
        # 1. 带通滤波
        emg_filtered = self.bandpass_filter(emg, lowcut=10, highcut=500)
        
        # 2. 整流
        emg_rectified = self.rectify_emg(emg_filtered)
        
        # 3. RMS平滑
        emg_smoothed = self.smooth_emg_rms(emg_rectified, 
                                           window_ms=200, step_ms=50)
        
        return emg_smoothed
    
    def preprocess_imu(self, imu):
        """
        IMU预处理流程
        
        步骤：
        1. 带通滤波 (10-500 Hz)
        2. 高通滤波去趋势 (0.5 Hz)
        """
        print("  处理IMU: 带通滤波 -> 去趋势")
        
        # 1. 带通滤波
        imu_filtered = self.bandpass_filter(imu, lowcut=10, highcut=500)
        
        # 2. 去趋势（去除重力分量）
        imu_detrended = self.highpass_filter(imu_filtered, cutoff=0.5)
        
        return imu_detrended
    
    def segment_data(self, emg, imu, labels, repetitions):
        """
        信号分割和窗口化
        
        参数:
            emg: EMG数据 (N, 12)
            imu: IMU数据 (N, 36)
            labels: 标签 (N,)
            repetitions: 重复索引 (N,)
            
        返回:
            segments: 字典包含训练/测试的EMG、IMU和标签
        """
        print(f"  分割数据: 窗口={self.window_size}ms, 步长={self.window_step}ms")
        
        train_emg, train_imu, train_labels = [], [], []
        test_emg, test_imu, test_labels = [], [], []
        
        # 识别运动边界（用于排除过渡期）
        label_changes = np.diff(labels, prepend=labels[0])
        transition_mask = np.zeros(len(labels), dtype=bool)
        
        # 标记运动边界前后的过渡期
        change_indices = np.where(label_changes != 0)[0]
        for idx in change_indices:
            start = max(0, idx - self.transition_margin)
            end = min(len(labels), idx + self.transition_margin)
            transition_mask[start:end] = True
        
        print(f"    排除过渡样本数: {np.sum(transition_mask)} / {len(labels)}")
        
        # 滑动窗口分割
        n_samples = emg.shape[0]
        for start in range(0, n_samples - self.window_samples + 1, self.step_samples):
            end = start + self.window_samples
            
            # 检查窗口是否包含过渡期
            if np.any(transition_mask[start:end]):
                continue
            
            # 提取窗口数据
            emg_window = emg[start:end, :]  # (400, 12)
            imu_window = imu[start:end, :]  # (400, 36)
            
            # 窗口标签：使用中心点的标签
            center_idx = start + self.window_samples // 2
            window_label = labels[center_idx]
            window_rep = repetitions[center_idx]
            
            # 分配到训练集或测试集
            if window_rep in self.train_reps:
                train_emg.append(emg_window)
                train_imu.append(imu_window)
                train_labels.append(window_label)
            elif window_rep in self.test_reps:
                test_emg.append(emg_window)
                test_imu.append(imu_window)
                test_labels.append(window_label)
        
        segments = {
            'train': {
                'emg': np.array(train_emg),
                'imu': np.array(train_imu),
                'labels': np.array(train_labels)
            },
            'test': {
                'emg': np.array(test_emg),
                'imu': np.array(test_imu),
                'labels': np.array(test_labels)
            }
        }
        
        print(f"    训练样本: {len(train_labels)}, 测试样本: {len(test_labels)}")
        
        return segments
    
    def normalize_data(self, train_data, test_data):
        """
        Z-score标准化
        
        使用训练集的均值和标准差对训练集和测试集进行标准化
        
        参数:
            train_data: 训练数据 (N_train, T, C)
            test_data: 测试数据 (N_test, T, C)
            
        返回:
            train_normalized: 标准化后的训练数据
            test_normalized: 标准化后的测试数据
            scaler: StandardScaler对象
        """
        n_train, T, C = train_data.shape
        n_test = test_data.shape[0]
        
        # 重塑为 (N*T, C) 用于拟合
        train_reshaped = train_data.reshape(-1, C)
        test_reshaped = test_data.reshape(-1, C)
        
        # 逐通道标准化
        scaler = StandardScaler()
        train_normalized = scaler.fit_transform(train_reshaped)
        test_normalized = scaler.transform(test_reshaped)
        
        # 重塑回 (N, T, C)
        train_normalized = train_normalized.reshape(n_train, T, C)
        test_normalized = test_normalized.reshape(n_test, T, C)
        
        return train_normalized, test_normalized, scaler
    
    
    def process_all_subjects(self):
        """
        处理所有受试者的数据（分别存储每个受试者）
        
        返回:
            processed_data: 包含每个受试者数据的字典
        """
        # 存储每个受试者的数据
        subjects_data = {}
        
        # 对每个受试者进行处理
        for subject in self.subjects:
            print(f"\n{'='*60}")
            print(f"处理受试者 S{subject}")
            print(f"{'='*60}")
            
            # 加载数据
            emg_list, imu_list, label_list, rep_list = self.load_subject_data(subject)
            
            # 存储当前受试者的数据
            subject_train_emg, subject_train_imu, subject_train_labels = [], [], []
            subject_test_emg, subject_test_imu, subject_test_labels = [], [], []
            subject_train_exercises, subject_test_exercises = [], []  # 新增：记录exercise
            
            # 处理每个练习
            for ex_idx, (emg, imu, labels, reps) in enumerate(
                zip(emg_list, imu_list, label_list, rep_list), 1
            ):
                print(f"\n练习 E{ex_idx}:")
                
                # EMG预处理
                emg_processed = self.preprocess_emg(emg)
                
                # IMU预处理
                imu_processed = self.preprocess_imu(imu)
                
                # 信号分割和窗口化
                segments = self.segment_data(emg_processed, imu_processed, 
                                            labels, reps)
                
                # 累积到当前受试者的数据中
                subject_train_emg.append(segments['train']['emg'])
                subject_train_imu.append(segments['train']['imu'])
                subject_train_labels.append(segments['train']['labels'])
                # 记录exercise信息（1=E1, 2=E2, 3=E3）
                subject_train_exercises.append(np.full(len(segments['train']['labels']), ex_idx, dtype=np.int32))
                
                subject_test_emg.append(segments['test']['emg'])
                subject_test_imu.append(segments['test']['imu'])
                subject_test_labels.append(segments['test']['labels'])
                subject_test_exercises.append(np.full(len(segments['test']['labels']), ex_idx, dtype=np.int32))
            
            # 合并当前受试者的所有练习
            subject_train_emg = np.concatenate(subject_train_emg, axis=0)
            subject_train_imu = np.concatenate(subject_train_imu, axis=0)
            subject_train_labels = np.concatenate(subject_train_labels, axis=0)
            subject_train_exercises = np.concatenate(subject_train_exercises, axis=0)  # 新增
            
            subject_test_emg = np.concatenate(subject_test_emg, axis=0)
            subject_test_imu = np.concatenate(subject_test_imu, axis=0)
            subject_test_labels = np.concatenate(subject_test_labels, axis=0)
            subject_test_exercises = np.concatenate(subject_test_exercises, axis=0)  # 新增
            
            print(f"\n受试者 S{subject} 数据统计:")
            print(f"  训练样本: {len(subject_train_labels)}")
            print(f"  测试样本: {len(subject_test_labels)}")
            
            # 归一化（每个受试者独立归一化）
            print(f"  归一化受试者 S{subject} 数据...")
            subject_train_emg_norm, subject_test_emg_norm, subject_emg_scaler = self.normalize_data(
                subject_train_emg, subject_test_emg
            )
            subject_train_imu_norm, subject_test_imu_norm, subject_imu_scaler = self.normalize_data(
                subject_train_imu, subject_test_imu
            )
            
            # 保存受试者数据
            subjects_data[subject] = {
                'train': {
                    'emg': subject_train_emg_norm,
                    'imu': subject_train_imu_norm,
                    'labels': subject_train_labels,
                    'exercises': subject_train_exercises  # 新增
                },
                'test': {
                    'emg': subject_test_emg_norm,
                    'imu': subject_test_imu_norm,
                    'labels': subject_test_labels,
                    'exercises': subject_test_exercises  # 新增
                },
                'scalers': {
                    'emg': subject_emg_scaler,
                    'imu': subject_imu_scaler
                }
            }
        
        # 组织最终数据
        processed_data = {
            'subjects_data': subjects_data,  # 每个受试者的独立数据
            'metadata': {
                'subjects': self.subjects,
                'fs': self.fs,
                'window_size_ms': self.window_size,
                'window_step_ms': self.window_step,
                'n_emg_channels': 12,
                'n_imu_channels': 36,
                'n_classes': 50,  # 固定50个类别
                'class_names': self._get_class_names()
            }
        }
        
        return processed_data
    
    def _get_class_names(self):
        """返回50个类别的名称（简化版）"""
        # 这里返回简化的类别标识
        return [f"Gesture_{i}" if i > 0 else "Rest" for i in range(50)]
    
    def save_processed_data(self, processed_data, output_dir):
        """
        保存处理后的数据（HDF5格式）- 分别保存每个受试者
        
        参数:
            processed_data: 处理后的数据字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("保存处理后的数据（HDF5格式）")
        print(f"{'='*60}")
        
        # 保存每个受试者的数据
        print("\n保存每个受试者的独立数据:")
        print("-" * 60)
        
        subjects_scalers = {}
        for subject, subject_data in processed_data['subjects_data'].items():
            # 保存训练集
            train_file = output_path / f'S{subject}_train.h5'
            with h5py.File(train_file, 'w') as f:
                f.create_dataset('emg', data=subject_data['train']['emg'], 
                               compression='gzip', compression_opts=4)
                f.create_dataset('imu', data=subject_data['train']['imu'], 
                               compression='gzip', compression_opts=4)
                f.create_dataset('labels', data=subject_data['train']['labels'], 
                               compression='gzip', compression_opts=4)
                f.create_dataset('exercises', data=subject_data['train']['exercises'],  # 新增
                               compression='gzip', compression_opts=4)
                # 保存属性
                f.attrs['subject'] = subject
                f.attrs['n_samples'] = len(subject_data['train']['labels'])
                f.attrs['window_size'] = subject_data['train']['emg'].shape[1]
                f.attrs['n_emg_channels'] = subject_data['train']['emg'].shape[2]
                f.attrs['n_imu_channels'] = subject_data['train']['imu'].shape[2]
            
            # 保存测试集
            test_file = output_path / f'S{subject}_test.h5'
            with h5py.File(test_file, 'w') as f:
                f.create_dataset('emg', data=subject_data['test']['emg'], 
                               compression='gzip', compression_opts=4)
                f.create_dataset('imu', data=subject_data['test']['imu'], 
                               compression='gzip', compression_opts=4)
                f.create_dataset('labels', data=subject_data['test']['labels'], 
                               compression='gzip', compression_opts=4)
                f.create_dataset('exercises', data=subject_data['test']['exercises'],  # 新增
                               compression='gzip', compression_opts=4)
                # 保存属性
                f.attrs['subject'] = subject
                f.attrs['n_samples'] = len(subject_data['test']['labels'])
                f.attrs['window_size'] = subject_data['test']['emg'].shape[1]
                f.attrs['n_emg_channels'] = subject_data['test']['emg'].shape[2]
                f.attrs['n_imu_channels'] = subject_data['test']['imu'].shape[2]
            
            print(f"受试者 S{subject}:")
            print(f"  训练: {train_file.name} ({len(subject_data['train']['labels'])} 样本)")
            print(f"  测试: {test_file.name} ({len(subject_data['test']['labels'])} 样本)")
            
            # 保存每个受试者的scaler
            subjects_scalers[subject] = subject_data['scalers']
        
        # 保存metadata和scalers
        metadata_file = output_path / 'metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'subjects_scalers': subjects_scalers,  # 每个受试者的scaler
                'metadata': processed_data['metadata']
            }, f)
        
        print(f"\n元数据已保存: {metadata_file.name}")
        print(f"\n提示: 使用 S{{subject}}_train.h5 和 S{{subject}}_test.h5 进行单受试者分析")
    
    def print_dataset_info(self, processed_data):
        """
        打印数据集基本信息
        
        参数:
            processed_data: 处理后的数据字典
        """
        print(f"\n{'='*60}")
        print("数据集基本信息")
        print(f"{'='*60}")
        
        metadata = processed_data['metadata']
        
        print(f"\n受试者: {metadata['subjects']}")
        print(f"采样率: {metadata['fs']} Hz")
        print(f"窗口大小: {metadata['window_size_ms']} ms")
        print(f"窗口步长: {metadata['window_step_ms']} ms")
        print(f"EMG通道数: {metadata['n_emg_channels']}")
        print(f"IMU通道数: {metadata['n_imu_channels']}")
        print(f"类别数: {metadata['n_classes']}")
        
        # 每个受试者的统计
        print(f"\n{'='*60}")
        print("各受试者数据统计")
        print(f"{'='*60}")
        
        for subject in metadata['subjects']:
            subject_data = processed_data['subjects_data'][subject]
            n_train = len(subject_data['train']['labels'])
            n_test = len(subject_data['test']['labels'])
            
            print(f"\n受试者 S{subject}:")
            print(f"  训练样本: {n_train}")
            print(f"  测试样本: {n_test}")
            print(f"  总计: {n_train + n_test}")
            
            # 类别分布
            train_labels = subject_data['train']['labels']
            test_labels = subject_data['test']['labels']
            unique_train = len(np.unique(train_labels))
            unique_test = len(np.unique(test_labels))
            print(f"  训练集类别数: {unique_train}")
            print(f"  测试集类别数: {unique_test}")
        
        print(f"\n{'='*60}\n")


def main():
    """主函数"""
    # 设置路径
    data_root = "/home/mlsnrs/data/wrj/MoMo/z重新整理的代码"
    output_dir = "/home/mlsnrs/data/wrj/MoMo/Momo/processed_data"
    
    # 创建预处理器（只处理受试者10）
    preprocessor = NinaproDB2Preprocessor(
        data_root=data_root,
        subjects=[10],  # 只处理受试者10
        fs=2000
    )
    
    # 处理所有受试者数据
    processed_data = preprocessor.process_all_subjects()
    
    # 保存处理后的数据
    preprocessor.save_processed_data(processed_data, output_dir)
    
    # 打印数据集基本信息
    preprocessor.print_dataset_info(processed_data)
    
    print("数据预处理完成！")


if __name__ == "__main__":
    main()

