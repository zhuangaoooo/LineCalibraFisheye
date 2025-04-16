import numpy as np
import cv2
from scipy.optimize import minimize

class FisheyeCalibrator:
    def __init__(self, classes, image_size):
        self.annotations = classes
        self.image_size = image_size
        self.K = None
        self.D = None
        self.optimization_steps = []

    @staticmethod
    def calculate_nonlinearity(points):
        """计算点集的非线性度指标（基于首尾端点拟合直线）"""
        points = np.array(points, dtype=np.float32)
        if len(points) < 2:
            raise ValueError("至少需要两个点进行直线拟合")
        
        p_start, p_end = points[0], points[-1]
        x1, y1 = p_start
        x2, y2 = p_end
        
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        
        denominator = np.sqrt(A**2 + B**2)
        if denominator == 0:
            return 0.0, (p_start, p_end)
        
        x, y = points[:, 0], points[:, 1]
        distances = np.abs(A*x + B*y + C) / denominator
        
        full_scale = max(np.ptp(x), np.ptp(y)) 
        max_deviation = np.max(distances)
        
        nonlinearity = (max_deviation / full_scale) * 100 if full_scale != 0 else 0.0
        return nonlinearity, (p_start, p_end)

    def _compute_total_nonlinearity(self, K, D):
        """计算所有类别的总非线性度"""
        total = 0.0
        for cls in self.annotations:
            undist_points = []
            for (x, y) in cls['points']:
                distorted_point = np.array([[[float(x), float(y)]]], dtype=np.float32)
                undistorted_points = cv2.fisheye.undistortPoints(
                    distorted_point, K, D, P=K)
                x_undist, y_undist = undistorted_points[0][0]
                undist_points.append((x_undist, y_undist))
            nonlinearity, _ = self.calculate_nonlinearity(undist_points)
            total += nonlinearity
        return total

    def optimize_step1(self):
        """第一步：优化焦距f"""
        w, h = self.image_size
        initial_f = np.sqrt(w**2 + h**2) / 2
        initial_params = [initial_f]

        bounds = [(0, initial_params[0]*2.0)]

        result = self._run_optimization(
            objective=self._step1_objective,
            initial=initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            step_name="Step 1"
        )

        optimal_f = result.x[0]
        self.K = np.array([
            [optimal_f, 0, w//2],
            [0, optimal_f, h//2],
            [0, 0, 1]
        ], dtype=np.float32)
        self.D = np.zeros((4, 1), dtype=np.float32)
        
        return optimal_f

    def _step1_objective(self, params):
        """第一步优化的目标函数"""
        f = params[0]
        K = np.array([[f, 0, self.image_size[0]//2],
                    [0, f, self.image_size[1]//2],
                    [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
        return self._compute_total_nonlinearity(K, D)

    def optimize_step2(self):
        """第二步：优化fx,fy,cx,cy,k1"""
        w, h = self.image_size
        initial_params = [
            self.K[0, 0],  # 从第一步结果初始化fx
            self.K[1, 1],  # fy
            self.K[0, 2],  # cx
            self.K[1, 2],  # cy
            0.0  # k1初始值
        ]

        # 边界条件设置
        bounds = [
            (initial_params[0]*0.8, initial_params[0]*1.2),  # fx
            (initial_params[1]*0.8, initial_params[1]*1.2),  # fy
            (w*0.3, w*0.7),  # cx
            (h*0.3, h*0.7),  # cy
            (-0.5, 0.5)       # k1
        ]

        result = self._run_optimization(
            objective=self._step2_objective,
            initial=initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            step_name="Step 2"
        )

        self.K = np.array([
            [result.x[0], 0, result.x[2]],
            [0, result.x[1], result.x[3]],
            [0, 0, 1]
        ], dtype=np.float32)
        self.D = np.array([result.x[4], 0, 0, 0], dtype=np.float32).reshape(4, 1)
        self.optimization_steps.append(result)
        return result

    def _step2_objective(self, params):
        """第二步优化的目标函数"""
        fx, fy, cx, cy, k1 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([k1, 0, 0, 0], dtype=np.float32).reshape(4, 1)
        return self._compute_total_nonlinearity(K, D)

    def optimize_step3(self):
        """第三步：优化全部参数（fx,fy,cx,cy,k1-k4）"""
        initial_params = [
            self.K[0, 0],  # fx
            self.K[1, 1],  # fy
            self.K[0, 2],  # cx
            self.K[1, 2],  # cy
            self.D[0, 0],  # k1
            0.0, 0.0, 0.0  # k2-k4初始值
        ]

        # 边界条件设置
        bounds = [
            (initial_params[0]*0.8, initial_params[0]*1.2),  # fx
            (initial_params[1]*0.8, initial_params[1]*1.2),  # fy
            (self.K[0,2]*0.8, self.K[0,2]*1.2),  # cx
            (self.K[1,2]*0.8, self.K[1,2]*1.2),  # cy
            (-0.5, 0.5),  # k1
            (-0.3, 0.3),  # k2
            (-0.1, 0.1),  # k3
            (-0.05, 0.05) # k4
        ]

        result = self._run_optimization(
            objective=self._step3_objective,
            initial=initial_params,
            bounds=bounds,
            method='SLSQP',
            step_name="Step 3"
        )

        self.K = np.array([
            [result.x[0], 0, result.x[2]],
            [0, result.x[1], result.x[3]],
            [0, 0, 1]
        ], dtype=np.float32)
        self.D = np.array(result.x[4:8], dtype=np.float32).reshape(4, 1)
        self.optimization_steps.append(result)
        return result

    def _step3_objective(self, params):
        """第三步优化的目标函数"""
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([k1, k2, k3, k4], dtype=np.float32).reshape(4, 1)
        return self._compute_total_nonlinearity(K, D)

    def _run_optimization(self, objective, initial, bounds, method, step_name):
        """通用优化执行框架
        Args:
            objective: 目标函数对象
            initial: 初始参数数组
            bounds: 参数约束范围列表
            method: 优化算法名称
            step_name: 阶段名称标识
        Returns:
            OptimizeResult: 优化结果对象
        """
        print(f"\n{step_name} Optimization:")
        result = minimize(
            objective,
            initial,
            method=method,
            bounds=bounds if bounds else None,
            options={
                'maxiter': 100,
                'disp': True,
                'eps': 1e-3
            }
        )
        self._print_optimization_result(result, step_name)
        return result

    def _print_optimization_result(self, result, step_name):
        """统一格式化输出优化结果"""
        print(f"\n{step_name} Results:")
        if hasattr(result, 'success'):
            print(f"Status: {'Success' if result.success else 'Failed'}")
            print(f"Message: {result.message}")
        print(f"Minimum Nonlinearity: {result.fun:.2f}%")
        print("Optimized Parameters:")
        print(np.array2string(result.x, precision=4, suppress_small=True))