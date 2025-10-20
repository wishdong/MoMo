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
    """Ninapro数据集预处理器（支持DB2/DB3/DB5/DB7）"""
    
    def __init__(self, data_root, subjects=[10, 23, 36], fs=2000, dataset='DB2'):
        """
        初始化预处理器
        
        参数:
            data_root: 数据根目录
            subjects: 受试者列表
            fs: 采样率 (Hz)
            dataset: 数据集名称 ('DB2', 'DB3', 'DB5', 'DB7')
        """
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.fs = fs
        self.dataset = dataset
        
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
        
        # 根据数据集设置目录格式和练习数量
        self._setup_dataset_config()
        
        print(f"初始化预处理器 - 数据集: {self.dataset}")
        print(f"采样率: {self.fs} Hz")
        print(f"窗口大小: {self.window_size} ms ({self.window_samples} samples)")
        print(f"窗口步长: {self.window_step} ms ({self.step_samples} samples)")
    
    def _setup_dataset_config(self):
        """根据数据集类型设置配置"""
        if self.dataset == 'DB2':
            self.subject_prefix = 'DB2_s'  # DB2_s10
            self.file_prefix = 'S'          # S10_E1_A1.mat
            self.exercises = [1, 2, 3]
        elif self.dataset == 'DB3':
            self.subject_prefix = 'DB3_s'  # DB3_s2
            self.file_prefix = 'S'          # S2_E1_A1.mat
            self.exercises = [1, 2, 3]
        elif self.dataset == 'DB5':
            self.subject_prefix = 's'       # s1
            self.file_prefix = 'S'          # S1_E1_A1.mat
            self.exercises = [1, 2, 3]      # DB5包含E1, E2, E3
        elif self.dataset == 'DB7':
            self.subject_prefix = 'Subject_'  # Subject_10
            self.file_prefix = 'S'            # S10_E1_A1.mat
            self.exercises = [1, 2]           # DB7只有E1和E2
        else:
            raise ValueError(f"不支持的数据集: {self.dataset}")
        
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
        
        # 根据数据集类型构建目录路径
        subject_dir = self.data_root / f"{self.subject_prefix}{subject}"
        emg_list, imu_list, label_list, rep_list = [], [], [], []
        
        # 加载练习数据（根据数据集类型）
        for exercise in self.exercises:
            mat_file = subject_dir / f"{self.file_prefix}{subject}_E{exercise}_A1.mat"
            
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
        print("  处理EMG: ", end='', flush=True)
        
        # 1. 带通滤波
        print("带通滤波...", end='', flush=True)
        emg_filtered = self.bandpass_filter(emg, lowcut=10, highcut=500)
        
        # 2. 整流
        print("整流...", end='', flush=True)
        emg_rectified = self.rectify_emg(emg_filtered)
        
        # 3. RMS平滑
        print("RMS平滑...", end='', flush=True)
        emg_smoothed = self.smooth_emg_rms(emg_rectified, 
                                           window_ms=200, step_ms=50)
        print("完成", flush=True)
        
        return emg_smoothed
    
    def preprocess_imu(self, imu):
        """
        IMU预处理流程
        
        步骤：
        1. 带通滤波 (10-500 Hz)
        2. 高通滤波去趋势 (0.5 Hz)
        """
        print("  处理IMU: ", end='', flush=True)
        
        # 1. 带通滤波
        print("带通滤波...", end='', flush=True)
        imu_filtered = self.bandpass_filter(imu, lowcut=10, highcut=500)
        
        # 2. 去趋势（去除重力分量）
        print("去趋势...", end='', flush=True)
        imu_detrended = self.highpass_filter(imu_filtered, cutoff=0.5)
        print("完成", flush=True)
        
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
        print(f"  分割数据: 窗口={self.window_size}ms, 步长={self.window_step}ms", end='', flush=True)
        
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
        
        print(f" (排除过渡样本: {np.sum(transition_mask)}/{len(labels)})", end='', flush=True)
        
        # 滑动窗口分割
        n_samples = emg.shape[0]
        total_windows = (n_samples - self.window_samples) // self.step_samples + 1
        print(f" 预计窗口数: {total_windows}...", end='', flush=True)
        
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
        
        print(f" 完成 (训练:{len(train_labels)}, 测试:{len(test_labels)})", flush=True)
        
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
    
    
    def process_single_subject(self, subject, output_dir):
        """
        处理单个受试者的数据并立即保存
        
        参数:
            subject: 受试者编号
            output_dir: 输出目录
            
        返回:
            success: 是否处理成功
        """
        try:
            print(f"\n{'='*60}")
            print(f"处理受试者 S{subject} [{self.dataset}]")
            print(f"{'='*60}")
            
            # 加载数据
            emg_list, imu_list, label_list, rep_list = self.load_subject_data(subject)
            
            if not emg_list:
                print(f"  错误: 受试者 S{subject} 没有有效数据")
                return False
            
            # 存储当前受试者的数据
            subject_train_emg, subject_train_imu, subject_train_labels = [], [], []
            subject_test_emg, subject_test_imu, subject_test_labels = [], [], []
            subject_train_exercises, subject_test_exercises = [], []
            
            # 处理每个练习
            for ex_idx, (emg, imu, labels, reps) in enumerate(
                zip(emg_list, imu_list, label_list, rep_list), 1
            ):
                print(f"\n练习 E{self.exercises[ex_idx-1]}:")
                
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
                subject_train_exercises.append(np.full(len(segments['train']['labels']), 
                                                      self.exercises[ex_idx-1], dtype=np.int32))
                
                subject_test_emg.append(segments['test']['emg'])
                subject_test_imu.append(segments['test']['imu'])
                subject_test_labels.append(segments['test']['labels'])
                subject_test_exercises.append(np.full(len(segments['test']['labels']), 
                                                     self.exercises[ex_idx-1], dtype=np.int32))
            
            # 合并当前受试者的所有练习
            subject_train_emg = np.concatenate(subject_train_emg, axis=0)
            subject_train_imu = np.concatenate(subject_train_imu, axis=0)
            subject_train_labels = np.concatenate(subject_train_labels, axis=0)
            subject_train_exercises = np.concatenate(subject_train_exercises, axis=0)
            
            subject_test_emg = np.concatenate(subject_test_emg, axis=0)
            subject_test_imu = np.concatenate(subject_test_imu, axis=0)
            subject_test_labels = np.concatenate(subject_test_labels, axis=0)
            subject_test_exercises = np.concatenate(subject_test_exercises, axis=0)
            
            print(f"\n受试者 S{subject} 数据统计:")
            print(f"  训练样本: {len(subject_train_labels)}")
            print(f"  测试样本: {len(subject_test_labels)}")
            
            # 归一化（每个受试者独立归一化）
            print(f"  归一化数据: ", end='', flush=True)
            print(f"EMG...", end='', flush=True)
            subject_train_emg_norm, subject_test_emg_norm, subject_emg_scaler = self.normalize_data(
                subject_train_emg, subject_test_emg
            )
            print(f"IMU...", end='', flush=True)
            subject_train_imu_norm, subject_test_imu_norm, subject_imu_scaler = self.normalize_data(
                subject_train_imu, subject_test_imu
            )
            print(f"完成", flush=True)
            
            # 立即保存当前受试者的数据
            self.save_single_subject(
                subject=subject,
                train_emg=subject_train_emg_norm,
                train_imu=subject_train_imu_norm,
                train_labels=subject_train_labels,
                train_exercises=subject_train_exercises,
                test_emg=subject_test_emg_norm,
                test_imu=subject_test_imu_norm,
                test_labels=subject_test_labels,
                test_exercises=subject_test_exercises,
                emg_scaler=subject_emg_scaler,
                imu_scaler=subject_imu_scaler,
                output_dir=output_dir
            )
            
            print(f"✓ 受试者 S{subject} 处理完成！\n")
            return True
            
        except Exception as e:
            print(f"✗ 受试者 S{subject} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_subjects(self, output_dir):
        """
        处理所有受试者的数据（逐个处理并保存）
        
        参数:
            output_dir: 输出目录
        """
        print(f"\n{'='*60}")
        print(f"批量处理 {self.dataset} 数据集")
        print(f"受试者列表: {self.subjects}")
        print(f"总计: {len(self.subjects)} 个受试者")
        print(f"{'='*60}")
        
        success_count = 0
        failed_subjects = []
        
        for idx, subject in enumerate(self.subjects, 1):
            print(f"\n[{idx}/{len(self.subjects)}] ", end='', flush=True)
            if self.process_single_subject(subject, output_dir):
                success_count += 1
            else:
                failed_subjects.append(subject)
        
        print(f"\n{'='*60}")
        print(f"批量处理完成！")
        print(f"成功: {success_count}/{len(self.subjects)}")
        if failed_subjects:
            print(f"失败的受试者: {failed_subjects}")
        print(f"{'='*60}\n")
    
    def _get_class_names(self):
        """返回50个类别的名称（简化版）"""
        # 这里返回简化的类别标识
        return [f"Gesture_{i}" if i > 0 else "Rest" for i in range(50)]
    
    def save_single_subject(self, subject, train_emg, train_imu, train_labels, train_exercises,
                           test_emg, test_imu, test_labels, test_exercises,
                           emg_scaler, imu_scaler, output_dir):
        """
        保存单个受试者的数据（HDF5格式）
        
        参数:
            subject: 受试者编号
            train_emg, train_imu, train_labels, train_exercises: 训练数据
            test_emg, test_imu, test_labels, test_exercises: 测试数据
            emg_scaler, imu_scaler: 归一化scaler
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存训练集
        print(f"  保存文件: ", end='', flush=True)
        train_file = output_path / f'S{subject}_train.h5'
        with h5py.File(train_file, 'w') as f:
            f.create_dataset('emg', data=train_emg, compression='gzip', compression_opts=4)
            f.create_dataset('imu', data=train_imu, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=train_labels, compression='gzip', compression_opts=4)
            f.create_dataset('exercises', data=train_exercises, compression='gzip', compression_opts=4)
            # 保存属性
            f.attrs['dataset'] = self.dataset
            f.attrs['subject'] = subject
            f.attrs['n_samples'] = len(train_labels)
            f.attrs['window_size'] = train_emg.shape[1]
            f.attrs['n_emg_channels'] = train_emg.shape[2]
            f.attrs['n_imu_channels'] = train_imu.shape[2]
        
        # 保存测试集
        test_file = output_path / f'S{subject}_test.h5'
        with h5py.File(test_file, 'w') as f:
            f.create_dataset('emg', data=test_emg, compression='gzip', compression_opts=4)
            f.create_dataset('imu', data=test_imu, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=test_labels, compression='gzip', compression_opts=4)
            f.create_dataset('exercises', data=test_exercises, compression='gzip', compression_opts=4)
            # 保存属性
            f.attrs['dataset'] = self.dataset
            f.attrs['subject'] = subject
            f.attrs['n_samples'] = len(test_labels)
            f.attrs['window_size'] = test_emg.shape[1]
            f.attrs['n_emg_channels'] = test_emg.shape[2]
            f.attrs['n_imu_channels'] = test_imu.shape[2]
        
        print(f"训练集({len(train_labels)}样本) + 测试集({len(test_labels)}样本) ✓", flush=True)
        
        # 保存scaler到metadata文件（追加模式）
        metadata_file = output_path / 'metadata.pkl'
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
        else:
            metadata = {
                'subjects_scalers': {},
                'metadata': {
                    'dataset': self.dataset,
                    'fs': self.fs,
                    'window_size_ms': self.window_size,
                    'window_step_ms': self.window_step,
                    'n_emg_channels': 12,
                    'n_imu_channels': 36,
                    'n_classes': 50
                }
            }
        
        # 更新scaler信息
        metadata['subjects_scalers'][subject] = {
            'emg': emg_scaler,
            'imu': imu_scaler
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Ninapro数据集预处理（支持DB2/DB3/DB5/DB7）')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['DB2', 'DB3', 'DB5', 'DB7'],
                       help='数据集名称')
    parser.add_argument('--subjects', type=str, required=True,
                       help='受试者列表，用逗号分隔（如: 1,2,3 或 1-10）')
    parser.add_argument('--data_root', type=str, 
                       default='/home/xuweishi/KBS25/MoMo/Momo/data',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/xuweishi/KBS25/MoMo/Momo/processed_data',
                       help='输出目录')
    parser.add_argument('--fs', type=int, default=2000,
                       help='采样率 (Hz)')
    
    args = parser.parse_args()
    
    # 解析受试者列表
    subjects = []
    for part in args.subjects.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    
    # 设置输出路径（按数据集分类）
    output_dir = Path(args.output_dir) / args.dataset
    
    print(f"\n{'='*60}")
    print(f"Ninapro {args.dataset} 数据预处理")
    print(f"{'='*60}")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {output_dir}")
    print(f"受试者: {subjects}")
    print(f"采样率: {args.fs} Hz")
    print(f"{'='*60}\n")
    
    # 创建预处理器
    preprocessor = NinaproDB2Preprocessor(
        data_root=Path(args.data_root) / args.dataset,
        subjects=subjects,
        fs=args.fs,
        dataset=args.dataset
    )
    
    # 处理所有受试者数据（逐个处理并保存）
    preprocessor.process_all_subjects(output_dir)
    
    print("\n数据预处理完成！")


if __name__ == "__main__":
    main()

