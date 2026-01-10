"""
Data reshaping and grid generation module.

Transforms unstructured 1D point data into structured 2D grids suitable for 3D surface plotting.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from .data_models import BoundaryQuantitiesData


def _remove_backtracking_spurs(r, z, v):
    """
    通过检测路径中的"回溯跳跃"去除中间的冗余分支（Spurs）。
    Greedy排序在遇到岔路时会走进死胡同，并在尽头"跳"回主路。
    此函数识别这种大跳跃，并剪除整个死胡同线段。
    
    改进版：
    1. 先行消除单点尖刺 (Smoothing)。
    2. 使用局部梯度/峰值检测识别跳跃，而非仅靠全局阈值。
    3. 寻找最佳回接点 (Branch Point) 剪除环路。
    
    Parameters
    ----------
    r, z, v : np.ndarray
        已按最近邻排序的点序列
        
    Returns
    -------
    Cleaned r, z, v
    """
    if len(r) < 10:
        return r, z, v


    # --- 1.5 预处理：消除多点局部回路 (Small Loop / Sharp Spike Removal) ---
    # 针对 "数个点" 构成的尖峰/回路：从分支点出发，走若干步后又回到分支点附近。
    # 例如: A -> B -> C -> D -> E，其中 A 和 E 距离非常近 (Loop闭合)
    # 此时应直接切断 B-C-D，保留 A-E 连接。
    
    if len(r) > 10:
        # 重新计算基础步长 stats
        dr = np.diff(r)
        dz = np.diff(z)
        steps = np.sqrt(dr**2 + dz**2)
        global_median_step = np.median(steps)
        if global_median_step == 0: global_median_step = 1e-6
        
        search_window = 25  # 向后搜索的窗口大小（覆盖“数个点”的尖峰）
        loop_tol_factor = 2.0  # 定义“邻居”的距离容忍度 (倍数于局部步长)
        
        keep_mask = np.ones(len(r), dtype=bool)
        i = 0
        while i < len(r) - 2:
            # 获取局部步长参考 (使用当前点的步长，或者全局中值兜底)
            # 注意：如果处于极密区域，步长可能很小；如果是稀疏区，步长很大。
            current_step = steps[i] if i < len(steps) else global_median_step
            # 阈值：只要闭合距离小于 2 倍正常步长，就认为是回到原点了 (Loop Closed)
            # 取 max 避免在步长极小的区域过于敏感
            threshold = max(current_step, global_median_step) * loop_tol_factor
            threshold_sq = threshold**2
            
            shortcut_found = False
            best_shortcut_idx = -1
            
            # 向后搜索寻找“回归点”
            # limit search range
            max_search = min(i + search_window, len(r))
            
            # 从远到近搜？还是从近到远？
            # 我们想要切掉最大的 Loop，所以理论上应该找最远的那个“近邻”。
            # 但是为了算法简单，我们遍历 range
            for j in range(i + 2, max_search):
                d_sq = (r[i] - r[j])**2 + (z[i] - z[j])**2
                if d_sq < threshold_sq:
                    best_shortcut_idx = j
                    shortcut_found = True
                    # 继续向后看有没有更远的回归点 (贪心：切最大的包)
            
            if shortcut_found:
                # 发现回路！
                # P[i] 和 P[best_shortcut_idx] 是邻居。
                # 剪掉中间的 P[i+1] ... P[best_shortcut_idx-1]
                keep_mask[i+1 : best_shortcut_idx] = False
                
                # 更新 i 指针，直接跳到回归点
                i = best_shortcut_idx
            else:
                i += 1
                
        # 应用剔除
        r = r[keep_mask]
        z = z[keep_mask]
        v = v[keep_mask]

    # --- 2. 识别回溯跳跃 (Main Spur Removal) ---
    # 重新计算步长
    dr = np.diff(r)
    dz = np.diff(z)
    steps = np.sqrt(dr**2 + dz**2)
    
    median_step = np.median(steps)
    if median_step == 0: median_step = 1e-6
    
    # 使用局部峰值检测 (Local Peak Detection)
    # 跳跃特征：步长突然变大，然后突然变小 (Small -> Large -> Small)
    # steps[i] 是从 P[i] 到 P[i+1] 的步长。如果是跳跃，steps[i] 应该是个局部极大值。
    
    # 填充边界以便比较
    s_prev = np.r_[steps[0], steps[:-1]] # steps[i-1]
    s_next = np.r_[steps[1:], steps[-1]] # steps[i+1]
    
    # 设定跳跃阈值: 必须显著大于全局中值 (过滤掉正常的网格变化)
    abs_threshold = max(median_step * 5.0, 0.05)
    
    # 判定条件：
    # 1. 绝对值大
    # 2. 是局部峰值 (比前后都大至少2倍)
    jump_mask = (steps > abs_threshold) & \
                (steps > 2.0 * s_prev) & \
                (steps > 2.0 * s_next)
                
    jump_indices = np.where(jump_mask)[0]
    
    if len(jump_indices) == 0:
        return r, z, v
        
    # 构建保留掩码
    keep_mask = np.ones(len(r), dtype=bool)
    
    for jump_idx in jump_indices:
        # P[jump_idx] -> (JUMP) -> P[jump_idx+1]
        # P[jump_idx] 是死胡同的尽头。
        # P[jump_idx+1] 是回到主路的点。
        
        # 我们寻找 P[jump_idx+1] 在路径更早期的“真正邻居”。
        # 搜索范围：0 ... jump_idx-1
        if jump_idx < 2: continue # 没得回溯
            
        p_resume_r = r[jump_idx+1]
        p_resume_z = z[jump_idx+1]
        
        candidates_r = r[:jump_idx] # 只看跳跃点之前的
        candidates_z = z[:jump_idx]
        
        dists_sq = (candidates_r - p_resume_r)**2 + (candidates_z - p_resume_z)**2
        
        # 找到最近的早期点
        branch_idx = np.argmin(dists_sq)
        min_dist = np.sqrt(dists_sq[branch_idx])
        
        # 验证连接性：如果接回去的距离很短（合理步长），说明这是个闭环
        # 阈值：可以是中位步长的几倍，或者跟跳跃之前的正常步长相当
        # 这里稍微放宽一点，允许接回去的距离稍微大一点点，但不应是巨大跳跃
        valid_connection = min_dist < max(median_step * 3.0, abs_threshold * 0.5)
        
        if valid_connection:
            # 确认分支！
            # 主路: ... -> P[branch_idx]
            # 死胡同: -> P[branch_idx+1] ... -> P[jump_idx]
            # 回归: -> P[jump_idx+1] ...
            # 我们要连接 P[branch_idx] 和 P[jump_idx+1]，剪掉中间的死胡同。
            
            # 剪切范围: (branch_idx + 1) 到 jump_idx (包含)
            keep_mask[branch_idx+1 : jump_idx+1] = False
            
    return r[keep_mask], z[keep_mask], v[keep_mask]


def _remove_radial_outliers(r_sorted, z_sorted, v_sorted, c_r, c_z, threshold=1.1, is_circular=True):
    """
    去除极径方向上的突刺点（用于清除交叉处的冗余延伸线）。
    基于极坐标下的局部平滑性：如果一个点比其前后邻居的插值显著更远，则视为离群点。
    
    Parameters
    ----------
    ...
    is_circular : bool
        是否为闭合曲线。如果是 (True)，首尾点会被视为邻居处理。
        如果否 (False)，首尾点被视为开放端点，不进行离群检测（保留）。
    """
    if len(r_sorted) < 5:
        return r_sorted, z_sorted, v_sorted
        
    # 迭代几次以清除连续的坏点
    for _ in range(2):
        if len(r_sorted) < 5: break
            
        dists = np.sqrt((r_sorted - c_r)**2 + (z_sorted - c_z)**2)
        
        if is_circular:
            # 环形边界处理：首尾相连
            d_prev = np.roll(dists, 1)
            d_next = np.roll(dists, -1)
        else:
            # 开放边界处理：不循环
            d_prev = np.empty_like(dists)
            d_prev[1:] = dists[:-1]
            d_prev[0] = dists[0]
            
            d_next = np.empty_like(dists)
            d_next[:-1] = dists[1:]
            d_next[-1] = dists[-1]
        
        # 预测极径 (邻居的平均)
        d_pred = (d_prev + d_next) * 0.5
        
        # 识别突刺：实际距离显著大于预测距离
        # 阈值包含比例项和绝对项
        is_spike = dists > (d_pred * threshold + 0.02)
        
        # 如果不是闭合曲线，强制保护端点不被此算法删除
        # (端点由专门的 peel 函数处理)
        if not is_circular:
            is_spike[0] = False
            is_spike[-1] = False
        
        if np.sum(is_spike) == 0:
            break
            
        # 保留非突刺点
        mask = ~is_spike
        r_sorted = r_sorted[mask]
        z_sorted = z_sorted[mask]
        v_sorted = v_sorted[mask]
        
    return r_sorted, z_sorted, v_sorted


def _sort_by_nearest_neighbor(r, z, anchor_point=None):
    """
    使用最近邻算法（贪心策略）对点云进行排序。
    
    Parameters
    ----------
    anchor_point : tuple(float, float), optional
        强制指定的起始参考点 (r, z)。
        排序将从点云中距离该锚点最近的点开始。
        这对于确保跨截面的排序方向一致性至关重要。
    """
    n = len(r)
    if n < 2:
        return np.arange(n)

    # 1. 寻找起始点
    if anchor_point is not None:
         # 指定了锚点，找离锚点最近的点作为 Start
        ref_r, ref_z = anchor_point
        dists_to_anchor = (r - ref_r)**2 + (z - ref_z)**2
        start_idx = np.argmin(dists_to_anchor)
    else:
        # 未指定，默认找离重心最近的点 (对开放直线可能会从中间开始，慎用)
        c_r, c_z = np.mean(r), np.mean(z)
        dists_to_center = (r - c_r)**2 + (z - c_z)**2
        start_idx = np.argmin(dists_to_center)

    ordered_indices = [start_idx]
    remaining_indices = set(range(n))
    remaining_indices.remove(start_idx)
    
    current_idx = start_idx
    
    # 2. 贪心路径搜索
    while remaining_indices:
        # 获取当前点坐标
        cur_r, cur_z = r[current_idx], z[current_idx]
        
        # 策略升级：不仅看最近邻，还要看方向
        # 1. 找到绝对最近距离
        min_dist_sq = 1.0e20
        # 这一步仍遍历寻找最小值
        for cand_idx in remaining_indices:
            d_sq = (r[cand_idx] - cur_r)**2 + (z[cand_idx] - cur_z)**2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
        
        if min_dist_sq > 1.0e15: # 孤立点或异常
             if remaining_indices:
                 print(f"[Warning] 剩余点无法连接，最近邻距离过大: {min_dist_sq}")
             break
             
        # 2. 筛选候选集合 (Min Distance + 5% Tolerance)
        # 允许一定误差，以此纳入分叉点附近的多个候选
        tolerance_sq = min_dist_sq * (1.05**2)
        candidates = []
        for cand_idx in remaining_indices:
            d_sq = (r[cand_idx] - cur_r)**2 + (z[cand_idx] - cur_z)**2
            if d_sq <= tolerance_sq:
                candidates.append(cand_idx)
        
        best_idx = -1
        
        # 3. 决策：如果只有一个候选，直接选；如果有多个，结合方向判断
        if len(candidates) == 1:
            best_idx = candidates[0]
        elif len(ordered_indices) < 2:
            # 没有足够的历史点来确定方向，回退到选距离最近的
            # 在 candidates 中找 d_sq 最小的
            best_sub_d = 1.0e20
            for c in candidates:
                d = (r[c] - cur_r)**2 + (z[c] - cur_z)**2
                if d < best_sub_d:
                    best_sub_d = d
                    best_idx = c
        else:
            # --- 核心逻辑：方向优先 ---
            # 计算 "梯度方向" (即当前路径的前进方向)
            prev_idx = ordered_indices[-2]
            prev_r, prev_z = r[prev_idx], z[prev_idx]
            
            # 向量 v_in: P_prev -> P_curr
            vec_in_r = cur_r - prev_r
            vec_in_z = cur_z - prev_z
            angle_in = np.arctan2(vec_in_z, vec_in_r)
            
            # 在候选点中寻找 "逆时针方向角度最小" 的点
            # 计算每个候选向量 v_out (P_curr -> P_cand) 相对于 v_in 的逆时针夹角
            min_angle_diff = 100.0 # big number
            
            for cand_idx in candidates:
                vec_out_r = r[cand_idx] - cur_r
                vec_out_z = z[cand_idx] - cur_z
                angle_out = np.arctan2(vec_out_z, vec_out_r)
                
                # 计算相对夹角 delta (0 ~ 2pi)
                # 0 表示直行，0+ 表示向左偏(逆时针)，2pi- 表示向右偏(顺时针)
                diff = angle_out - angle_in
                while diff < 0: diff += 2*np.pi
                while diff >= 2*np.pi: diff -= 2*np.pi
                
                # 优先选 diff 最小的 (最靠左/最顺着逆时针趋势的)
                if diff < min_angle_diff:
                    min_angle_diff = diff
                    best_idx = cand_idx
           
        ordered_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        current_idx = best_idx
        
    return np.array(ordered_indices)


def _peel_outlier_tips(r, z, v, max_dist_factor=None):
    """
    从有序曲线的两端向内剥离（删除）过于平均距离的点。
    适用于去除密集的延伸尾部（Tail）。
    
    Parameters
    ----------
    max_dist_factor : float
        如果端点距离超过 (平均距离 * factor)，则视为延伸尾部并剥离。
        如果为 None，则自动根据分布估算。
    """
    if len(r) < 10:
        return r, z, v
    # 计算相邻步长
    dr = np.diff(r)
    dz = np.diff(z)
    steps = np.sqrt(dr**2 + dz**2)
    
    # 计算平均步长
    step_mean = np.mean(steps)
    if step_mean == 0: step_mean = 1e-6 # 防止除零
    
    
    # 设定截断阈值
    jump_threshold = max(step_mean * max_dist_factor, 0.05) # 绝对阈值保底
    
    # 剥离的部分为数组尾部。查找距离大于阈值的部分
    jump_indices = np.where(steps > jump_threshold)[0]
    # 从第一次跳跃就剥离掉尾部
    if len(jump_indices) > 0:
        first_jump_idx = jump_indices[0]
        r = r[:first_jump_idx+1]
        z = z[:first_jump_idx+1]
        v = v[:first_jump_idx+1]
    
    return r, z, v



def reshape_to_grid(
    block: np.ndarray,
    col_names: List[str],
    names: Tuple[str, str, str, str],
    iplane: Optional[int] = None,
    xpoints: Optional[np.ndarray] = None,
    debug: bool = False
) -> BoundaryQuantitiesData:
    """
    将非结构化的1D点数据重整化为结构化的2D网格 (Toroidal x Poloidal)。

    该函数按环向平面 (Phi) 分组原始数据，在每个切面上进行极向排序，
    并堆叠成适合3D表面绘图的2D网格。

    Parameters
    ----------
    block : numpy.ndarray
        原始输入数据矩阵，形状为 (N_samples, N_features)
    col_names : list of str
        列名列表
    names : tuple of 4 str
        物理量列映射，预期顺序：[R_name, Z_name, phi_name, val_name]
    iplane : int, optional
        环向平面数（用于重整化检查）
    xpoints : numpy.ndarray, optional
        形状为 (2, 2) 的X点坐标数组，用于复杂几何的分段排序
        如果提供，排序会根据上/下X点分段进行
        如果为None，使用标准的重心角度排序
    debug : bool, optional
        调试模式标志，默认False

    Returns
    -------
    BoundaryQuantitiesData
        包含重整化2D网格的数据对象

    Raises
    ------
    ValueError
        如果列名在col_names中找不到
    """
    
    # 1. 解析列名对应关系
    r_col_name, z_col_name, phi_col_name, val_col_name = names
    if debug:
        print(f"[Reshaping] Mapping columns: R='{r_col_name}', Z='{z_col_name}', phi='{phi_col_name}', value='{val_col_name}'")
        print(f"[Reshaping] Input block shape: {block.shape}")
        print(f"[Reshaping] Using iplane={iplane}, xpoints={'provided' if xpoints is not None else 'not provided'}")
    try:
        r_idx = col_names.index(r_col_name)
        z_idx = col_names.index(z_col_name)
        phi_idx = col_names.index(phi_col_name)
        val_idx = col_names.index(val_col_name)
    except ValueError as e:
        raise ValueError(f"Column name mismatch: {e}")

    if debug:
        print(f"[Reshaping] Column indices: R={r_idx}, Z={z_idx}, phi={phi_idx}, value={val_idx}")
    # 提取原始一维数据
    R_raw = block[:, r_idx]
    Z_raw = block[:, z_idx]
    phi_raw = block[:, phi_idx]
    val_raw = block[:, val_idx]

    # 2. 识别唯一的 Phi 切面 (Toroidal Planes)
    unique_phi = np.unique(np.round(phi_raw, 5))
    unique_phi.sort()
    # unique_phi = unique_phi - unique_phi[0]  # 使第一个切面为0
    
    n_phi = len(unique_phi)
    if debug:
        print(f"[Reshaping] Detected {n_phi} toroidal planes (phi slices).")

    # 容器：用于存放整理后的每一圈的数据
    R_slices = []
    Z_slices = []
    Phi_slices = []
    Val_slices = []
    points_per_slice = []

    # 3. 遍历每个 Phi 切面进行 R-Z 排序
    for current_phi in unique_phi:
        # 3.1 提取当前切面的所有点
        mask = np.abs(phi_raw - current_phi) < 1e-4
        
        if debug:
            print(f"[Reshaping] Processing phi={current_phi:.5f} with {np.sum(mask)} points.")
        
        if xpoints is not None:
            # X点分段排序（用于双X点撕裂模）
            c_rup, c_zup = xpoints[1, :]
            c_rdn, c_zdn = xpoints[0, :]
            mask_xpt_up = Z_raw[mask] >= 0
            mask_xpt_dn = Z_raw[mask] < 0
            
            r_slice_up = R_raw[mask][mask_xpt_up]
            z_slice_up = Z_raw[mask][mask_xpt_up]
            r_slice_dn = R_raw[mask][mask_xpt_dn]
            z_slice_dn = Z_raw[mask][mask_xpt_dn]
            
            
            # angles_up = np.arctan2(z_slice_up - c_zup, r_slice_up - c_rup)
            # angles_dn = np.arctan2(z_slice_dn - c_zdn, r_slice_dn - c_rdn)
            
            # angle_up_0 = np.arctan2(-c_zup, np.max(r_slice_up) - c_rup)
            # angle_down_0 = np.arctan2(-c_zdn, np.min(r_slice_dn) - c_rdn)
            
            # --- 使用最近邻排序替代角度排序 ---
            # 策略升级：根据用户指定的Z=0切面逻辑寻找锚点
            
            # 计算当前切面的平均半径，用于粗略区分 内侧(Inboard) 和 外侧(Outboard)
            c_r_all = np.mean(R_raw[mask])

            # Upper Section (Z >= 0):
            # 目标锚点：Z最小（接近0），且 R较大（位于外侧）
            # 路径：Outboard Midplane -> Top -> Inboard Midplane
            if len(r_slice_up) > 0:
                # 1. 筛选出位于外侧的点 (R > mean)
                mask_out = r_slice_up > c_r_all
                if np.any(mask_out):
                    # 2. 在外侧点中找 Z 最小的
                    idxs_out = np.where(mask_out)[0]
                    idx_best = idxs_out[np.argmin(z_slice_up[idxs_out])]
                    anchor_up = (r_slice_up[idx_best], z_slice_up[idx_best])
                else:
                    # Fallback
                    idx_max = np.argmax(r_slice_up)
                    anchor_up = (r_slice_up[idx_max], z_slice_up[idx_max])
                
                sort_idx_up = _sort_by_nearest_neighbor(R_raw[mask][mask_xpt_up], z_slice_up, anchor_point=anchor_up)
            else:
                sort_idx_up = np.array([], dtype=int)


            # Lower Section (Z < 0):
            # 目标锚点：Z最大（接近0），且 R较小（位于内侧） -> 为了能和 Upper 的结尾 (Inboard) 接上
            # 路径：Inboard Midplane -> Bottom -> Outboard Midplane
            if len(r_slice_dn) > 0:
                # 1. 筛选出位于内侧的点 (R < mean)
                mask_in = r_slice_dn < c_r_all
                if np.any(mask_in):
                    # 2. 在内侧点中找 Z 最大的 (即最接近 Z=0)
                    idxs_in = np.where(mask_in)[0]
                    idx_best = idxs_in[np.argmax(z_slice_dn[idxs_in])]
                    # 重要修复：我们希望从 Inboard 连到 Outboard
                    # 但贪心搜索是无方向的，它只会找最近的。
                    # 如果起点在 Inboard，它会自然向 Bottom 探索，最后到达 Outboard。
                    # 这与我们要接在 Upper(Out->Top->In) 后面是吻合的吗？
                    # Upper: Out -> Top -> In
                    # Lower: In -> Bottom -> Out
                    # 拼接: Out -> Top -> In -> Bottom -> Out (完美闭环)
                    anchor_dn = (r_slice_dn[idx_best], z_slice_dn[idx_best])
                else:
                    # Fallback
                    idx_min = np.argmin(r_slice_dn)
                    anchor_dn = (r_slice_dn[idx_min], z_slice_dn[idx_min])
                    
                sort_idx_dn = _sort_by_nearest_neighbor(r_slice_dn, z_slice_dn, anchor_point=anchor_dn)
            else:
                sort_idx_dn = np.array([], dtype=int)
            
             # 确保 Lower 的终点与 Upper 的起点接近 (闭合检查)
            if len(sort_idx_up) > 0 and len(sort_idx_dn) > 0:
                # Upper Start: Outboard Midplane
                # Upper End: Inboard Midplane
                # Lower Start: Inboard Midplane
                # Lower End: Outboard Midplane
                
                # Check 1: Upper End -> Lower Start
                # 理论上 Upper[-1] 应该和 Lower[0] 很近
                
                # Check 2: Lower End -> Upper Start
                # 理论上 Lower[-1] 应该和 Upper[0] 很近 (闭环)
                
                # 如果 Lower 的方向反了 (变成了 Out -> Bottom -> In)
                # 那么 Lower[0] 是 Outboard, 与 Upper[-1](Inboard) 距离很远
                # 这种情况下我们需要翻转 Lower
                
                r_u_end, z_u_end = r_slice_up[sort_idx_up][-1], z_slice_up[sort_idx_up][-1]
                r_d_start, z_d_start = r_slice_dn[sort_idx_dn][0], z_slice_dn[sort_idx_dn][0]
                r_d_end, z_d_end = r_slice_dn[sort_idx_dn][-1], z_slice_dn[sort_idx_dn][-1]
                
                dist_normal = (r_u_end - r_d_start)**2 + (z_u_end - z_d_start)**2
                dist_flipped = (r_u_end - r_d_end)**2 + (z_u_end - z_d_end)**2
                
                if dist_flipped < dist_normal:
                    if debug: print("[Reshaping] Auto-flipping Lower section to match boundary continuity.")
                    sort_idx_dn = sort_idx_dn[::-1]

            
            # --- 分别对上、下两部分进行清洗 ---
            
            # --- 分别对上、下两部分进行清洗 ---
            # 上半部分
            r_up = r_slice_up[sort_idx_up]
            z_up = z_slice_up[sort_idx_up]
            v_up = val_raw[mask][mask_xpt_up][sort_idx_up]
            
            # 1. 端点剥离 (去除路径末端可能残留的长尾)
            r_up, z_up, v_up = _peel_outlier_tips(
                 r_up, z_up, v_up,max_dist_factor=2.0
             )
            
            # 2. 剪除回溯分支 (去除中间的死胡同 - 基于跳跃)
            # 这是处理拓扑冗余(Spurs)最核心的步骤
            #r_up, z_up, v_up = _remove_backtracking_spurs(r_up, z_up, v_up)




            # 下半部分
            r_dn = r_slice_dn[sort_idx_dn]
            z_dn = z_slice_dn[sort_idx_dn]
            v_dn = val_raw[mask][mask_xpt_dn][sort_idx_dn]
            
            
            r_dn, z_dn, v_dn = _peel_outlier_tips(
                r_dn, z_dn, v_dn, max_dist_factor=2.0
            )
            
            #r_dn, z_dn, v_dn = _remove_backtracking_spurs(r_dn, z_dn, v_dn)
            

            
            # 合并上下部分
            # 连接顺序： Upper(Out->In) ... Lower(In->Out)
            # 这样自然形成闭环，无需翻转
            r_sorted = np.concatenate((r_up, r_dn))
            z_sorted = np.concatenate((z_up, z_dn))
            v_sorted = np.concatenate((v_up, v_dn))
            # --------------------------------
            
            R_slices.append(r_sorted)
            Z_slices.append(z_sorted)
            Phi_slices.append(np.full_like(r_sorted, current_phi))
            Val_slices.append(v_sorted)
            
            points_per_slice.append(len(r_sorted))
        
        else:
            # 标准重心角度排序
            r_slice = R_raw[mask]
            z_slice = Z_raw[mask]
            v_slice = val_raw[mask]
            
            if len(r_slice) == 0:
                continue
            
            # 计算重心
            c_r = np.mean(r_slice)
            c_z = np.mean(z_slice)
        
            if debug:
                print(f"[Reshaping] Phi={current_phi:.5f}: Centroid at (R={c_r:.3f}, Z={c_z:.3f}), Points={len(r_slice)}")
        
            # 计算每个点相对于重心的角度
            # 这里假定数据是单连通闭合曲线（如偏滤器靶板或第一壁截面）
            # 使用简单的重心极角排序可能对复杂非凸形状（如弯曲的偏滤器腿）产生问题
            # 但对于"凹凸多边形但大致呈环状"的结构通常有效
            angles = np.arctan2(z_slice - c_z, r_slice - c_r)
            sort_idx = np.argsort(angles)
            
            r_sorted = r_slice[sort_idx]
            z_sorted = z_slice[sort_idx]
            v_sorted = v_slice[sort_idx]
            
            # --- 新增：清洗交叉处的冗余突刺点 ---
            # 这种点通常特征是：在排序后的多边形路径上，出现了远离重心的尖峰
            r_sorted, z_sorted, v_sorted = _remove_radial_outliers(
                r_sorted, z_sorted, v_sorted, c_r, c_z, threshold=1.05
            )
            # --------------------------------
            
            R_slices.append(r_sorted)
            Z_slices.append(z_sorted)
            Phi_slices.append(np.full_like(r_sorted, current_phi))
            Val_slices.append(v_sorted)
            
            points_per_slice.append(len(r_sorted))

    # 4. 数据对齐检查
    if len(set(points_per_slice)) > 1:
        if debug:
            print(f"[Reshaping] Warning: Points per slice vary: {set(points_per_slice)}")
        min_points = min(points_per_slice)
        # 截断到最小公共大小
        for i in range(len(R_slices)):
            R_slices[i] = R_slices[i][:min_points]
            Z_slices[i] = Z_slices[i][:min_points]
            Phi_slices[i] = Phi_slices[i][:min_points]
            Val_slices[i] = Val_slices[i][:min_points]
    
    # 5. 堆叠成 2D 矩阵 (N_phi x N_poloidal)
    R_grid = np.vstack(R_slices)
    Z_grid = np.vstack(Z_slices)
    Phi_grid = np.vstack(Phi_slices)
    Val_grid = np.vstack(Val_slices)
    
    if debug:
        print(f"[Reshaping] Reshaped grid size: {R_grid.shape}")
    
    return BoundaryQuantitiesData(
        R=R_grid,
        Z=Z_grid,
        phi=Phi_grid,
        data=Val_grid,
        data_name=val_col_name,
        grid_shape=R_grid.shape
    )

