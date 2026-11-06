import numpy as np

def get_env(name):
    if name == 'm1': # nonadditive
        def f_x0(zs, ux0, z_coef=1, **kwargs):
            n = zs.shape[0]
            x0 = np.zeros((n, 2))

            # the first dimension
            gamma0_1 = np.array([-0.45, 1.5 * z_coef, 1, 0.5])
            M_1 = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                    ux0[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            x0_1 = np.cbrt(M_1 @ gamma0_1)
            x0[:, 0] = x0_1

            # the second dimension
            gamma0_2 = np.array([-0.75, 2.5 * z_coef, 1, 1, 0.5])
            M_2 = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                    ux0[:, 1].reshape(-1, 1),
                    ux0[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            x0_2 = np.cbrt(M_2 @ gamma0_2)
            x0[:, 1] = x0_2

            return x0
        
        def f_xt(zs, xtm1, atm1, uxt, z_coef=1, **kwargs):
            xtm1_1 = xtm1[:, 0].reshape(-1, 1)
            xtm1_2 = xtm1[:, 1].reshape(-1, 1)
            n = xtm1.shape[0]
            xt = np.zeros((n, 2))

            # the first dimension
            gamma_1 = np.array([-0.45, 0.3, 0.3 * z_coef, 0.45 * z_coef, 0.5, 0.25]) 
            M_1 = np.concatenate(
                [
                    np.ones([n, 1]),
                    xtm1_1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_1 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                    uxt[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            xt_1 = np.cbrt(M_1 @ gamma_1)
            xt[:, 0] = xt_1

            # the second dimension
            gamma_2 = np.array([-0.9, 0.4, 0.2, 0.6 * z_coef, 0.2 * z_coef, 0.9 * z_coef, 0.5, 0.9, 0.5])
            M_2 = np.concatenate(
                [
                    np.ones([n, 1]),
                    xtm1_1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_2 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_1 * (zs - 0.5),
                    xtm1_2 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                    uxt[:, 1].reshape(-1, 1),
                    uxt[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            xt_2 = np.cbrt(M_2 @ gamma_2)
            xt[:, 1] = xt_2

            return xt

        def f_rt(zs, xt, at, urt, z_coef=1, **kwargs):
            xt_1 = xt[:, 0].reshape(-1, 1)
            xt_2 = xt[:, 1].reshape(-1, 1)
            lmbda = np.array([-0.3, 0.2 * z_coef, 0.5, 0.2 * z_coef, 0.5, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt_1 * zs, xt_1 * at, xt_2 * zs, xt_2 * at, zs * at, urt], axis=1
            )
            rt = np.cbrt(M @ lmbda)
            return rt

    elif name == 'm2': # additive
        def f_x0(zs, ux0, z_coef=1, **kwargs):
            ux0 = ux0[:, 0].reshape(-1, 1)
            gamma0 = np.array([-0.3, 1 * z_coef, 1])
            n = zs.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0,
                ],
                axis=1,
            )
            x0 = M @ gamma0
            x0 = x0.reshape(-1, 1)
            return x0

        def f_xt(zs, xtm1, atm1, uxt, z_coef=1, **kwargs):
            uxt = uxt[:, 0].reshape(-1, 1)
            gamma = np.array([-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1])
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    (zs - 0.5),
                    xtm1,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt,
                ],
                axis=1,
            )
            xt = M @ gamma
            xt = xt.reshape(-1, 1)
            return xt

        def f_rt(zs, xt, at, urt, z_coef=1, **kwargs):
            lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, zs, at, xt * zs, xt * at, zs * at, urt], axis=1
            )
            rt = M @ lmbda
            return rt

    elif name == 'm3': # Environment with higher-dimensional state and sensitive attribute
        dim_x = 20
        
        gamma0_matrix = np.array([
                                    [-0.2,  0.4,  0.5,  0.3,  0.2],
                                    [-0.2,  0.4,  0.2,  0.5,  0.9],
                                    [-0.5,  0.4,  0.2,  0.1,  0.1],
                                    [-0.1,  0.2,  0.1,  0.6,  0.4],
                                    [-0.1,  0.5,  0.5,  0.4,  0.1],
                                    [-0.7,  0.3,  0.4,  0.1,  0.3],
                                    [-0.5,  0.3,  0.2,  0.5,  0.2],
                                    [-0.1,  0.3,  0.2,  0.1,  0.4],
                                    [-0.1,  0.1,  0.1,  0.4,  0.4],
                                    [-0.3,  0.2,  0.3,  0.2,  1.0],
                                    [-0.4,  0.4,  0.3,  0.4,  0.3],
                                    [-0.2,  0.2,  0.3,  0.4,  0.5],
                                    [-0.2,  0.5,  0.4,  0.3,  0.4],
                                    [-0.3,  0.5,  0.1,  0.3,  0.3],
                                    [-0.4,  0.3,  0.2,  0.1,  0.4],
                                    [-0.5,  0.6,  0.2,  0.3,  0.4],
                                    [-0.1,  0.2,  0.3,  0.2,  0.3],
                                    [-0.8,  0.3,  0.1,  0.2,  1.0],
                                    [-0.1,  0.2,  0.3,  0.5,  0.4],
                                    [-0.9,  0.1,  0.5,  0.1,  0.5]
                                ])
        
        gamma_matrix = np.array([
                            [-0.7,  0.5,  0.6,  0.2,  0.3,  0.7,  0.6,  0.4,  0.5,  0.2,  0.1,  0.2,  0.5,  0.6],
                            [-0.3,  0.2,  0.4,  0.5,  0.3,  0.5,  0.5,  0.2,  0.3,  0.1,  0.3,  0.1,  0.4,  0.2],
                            [-0.6,  0.4,  0.6,  0.6,  0.3,  0.3,  0.7,  0.1,  0.1,  0.4,  0.3,  0.7,  0.2,  0.5],
                            [-0.4,  0.6,  0.6,  0.7,  0.5,  0.3,  0.4,  0.1,  0.1,  0.2,  0.2,  0.2,  0.1,  0.4],
                            [-0.5,  0.2,  0.4,  0.5,  0.2,  0.3,  0.7,  0.3,  0.6,  0.4,  0.4,  0.1,  0.5,  0.1],
                            [-0.7,  0.2,  0.5,  0.6,  0.5,  0.6,  0.4,  0.1,  0.2,  0.5,  0.1,  0.3,  0.1,  0.1],
                            [-0.8,  0.6,  0.5,  0.7,  0.3,  0.4,  0.5,  0.6,  0.3,  0.5,  0.1,  0.2,  0.6,  0.4],
                            [-0.5,  0.6,  0.6,  0.4,  0.3,  0.4,  0.6,  0.1,  0.4,  0.1,  0.3,  0.3,  0.1,  0.8],
                            [-0.9,  0.6,  0.4,  0.5,  0.2,  0.7,  0.5,  0.4,  0.2,  0.3,  0.3,  0.2,  0.2,  0.3],
                            [-0.3,  0.5,  0.3,  0.4,  0.2,  0.5,  0.3,  0.7,  0.5,  0.4,  0.1,  0.2,  0.4,  0.1],
                            [-0.4,  0.2,  0.5,  0.6,  0.6,  0.5,  0.2,  0.1,  0.1,  0.1,  0.2,  0.2,  0.3,  0.5],
                            [-0.6,  0.3,  0.6,  0.7,  0.6,  0.6,  0.2,  0.5,  0.2,  0.3,  0.4,  0.2,  0.1,  0.2],
                            [-0.5,  0.4,  0.6,  0.6,  0.3,  0.3,  0.4,  0.2,  0.1,  0.5,  0.1,  0.1,  0.5,  0.3],
                            [-0.8,  0.2,  0.3,  0.7,  0.4,  0.4,  0.3,  0.3,  0.4,  0.4,  0.5,  0.6,  0.2,  0.7],
                            [-0.7,  0.4,  0.6,  0.4,  0.5,  0.3,  0.5,  0.1,  0.5,  0.1,  0.4,  0.2,  0.6,  0.2],
                            [-0.5,  0.2,  0.2,  0.6,  0.5,  0.4,  0.4, -0.4,  0.1,  0.2,  0.1,  0.3,  0.6,  0.1],
                            [-0.3,  0.4,  0.2,  0.6,  0.2,  0.4,  0.3,  0.4,  0.5,  0.4,  0.2,  0.2,  0.4,  0.5],
                            [-0.8,  0.4,  0.5,  0.3,  0.2,  0.3,  0.3,  0.4,  0.3,  0.5,  0.1,  0.3,  0.1,  0.3],
                            [-0.7,  0.2,  0.3,  0.5,  0.2,  0.4,  0.3,  0.1,  0.2,  0.1,  0.3,  0.1,  0.4,  0.6],
                            [-0.5,  0.4,  0.3,  0.7,  0.4,  0.3,  0.4,  0.4,  0.4,  0.5,  0.2,  0.2,  0.1,  0.4]
                        ]) 
        gamma_matrix[:, 5:7] = gamma_matrix[:, 5:7] - 0.04 
        
        lmbda = np.array([
                            -0.3,  0.4,  0.3,  0.4,  0.5,  0.2,  0.5,  0.2,  0.5,  0.3,  
                                0.4,  0.5,  0.3,  0.4,  0.3, -0.3,  0.2,  0.5,  0.5,  0.3,  
                                0.2,  0.5,  0.4,  0.4,  0.4,  0.1,  0.2, -0.3,  0.5,  0.4,  
                                0.5,  0.3,  0.2,  0.3,  0.4,  0.4,  0.3,  0.5,  0.5,  0.4,  
                                0.3,  0.1,  0.2,  0.4
                        ])
        lmbda[-2] = -1.0  
        lmbda[-3] = -0.6

        def f_x0(zs, ux0, v_coef=0, z_coef=1):
            n = zs.shape[0]
            x0 = np.zeros((n, dim_x))
            zs1 = zs[:, 0].reshape(-1, 1)
            zs2 = zs[:, 1].reshape(-1, 1)

            for i in range(dim_x):
                g = gamma0_matrix[i].copy()
                g[1] *= z_coef  # Scale the Z coefficient
                g[2] *= z_coef  # Scale the Z coefficient
                
                u_idx1, u_idx2 = i, -1
                
                M = np.concatenate([
                    np.ones([n, 1]),
                    zs1,
                    zs2,
                    ux0[:, u_idx1].reshape(-1, 1),
                    ux0[:, u_idx2].reshape(-1, 1)
                ], axis=1)
                
                x0[:, i] = np.cbrt(M @ g)
            return x0

        def f_xt(zs, xtm1, atm1, uxt, v_coef=0, z_coef=1):
            n = xtm1.shape[0]
            xt = np.zeros((n, dim_x))
            zs1_centered = (zs[:, 0].reshape(-1, 1) - 0.5)
            zs2_centered = (zs[:, 1].reshape(-1, 1) - 0.5)
            at_centered = (atm1.reshape(-1, 1) - 0.5)
            
            for i in range(dim_x):
                g = gamma_matrix[i].copy()
                g[2] *= z_coef  # Scale xtm1 * zs coefficient
                g[4] *= z_coef  # Scale xtm1 * zs coefficient
                g[5] *= z_coef  # Scale zs * at coefficient
                g[6] *= z_coef  # Scale zs * at coefficient
                g[7] *= z_coef
                g[8] *= z_coef
                
                xtm1_self = xtm1[:, i].reshape(-1, 1)
                xtm1_cross = xtm1[:, (i + 1) % dim_x].reshape(-1, 1)
                
                u_idx1, u_idx2 = i, -1
                
                M = np.concatenate([
                    np.ones([n, 1]),
                    xtm1_self * at_centered,
                    xtm1_self * zs2_centered,
                    xtm1_cross * at_centered,
                    xtm1_cross * zs1_centered,
                    zs1_centered * at_centered,
                    zs2_centered * at_centered,
                    zs1_centered, 
                    zs2_centered, 
                    at_centered, 
                    xtm1_self, 
                    xtm1_cross, 
                    uxt[:, u_idx1].reshape(-1, 1),
                    uxt[:, u_idx2].reshape(-1, 1), 
                ], axis=1)
                
                xt[:, i] = np.cbrt(M @ g)
            return xt

        def f_rt(zs, xt, at, urt, v_coef=0, z_coef=1):
            n = xt.shape[0]
            zs1_col = zs[:, 0].reshape(-1, 1)
            zs2_col = zs[:, 1].reshape(-1, 1)
            at_col = at.reshape(-1, 1)
            
            lmbda_scaled = lmbda.copy()
            
            lmbda_scaled[1] *= z_coef
            lmbda_scaled[2] *= z_coef
            lmbda_scaled[5] *= z_coef
            lmbda_scaled[6] *= z_coef
            lmbda_scaled[-3] *= z_coef
            lmbda_scaled[-2] *= z_coef

            lmbda_scaled = np.array([-0.6, 0.5 * z_coef, 0.5 * z_coef, 0.3, 0.4, 0.4 * z_coef, 0.4 * z_coef, 0.2, -0.7 * z_coef, -0.7 * z_coef, 1.0])

            xt_col = np.mean(xt, axis=1, keepdims=True)
            xt_zs_components = [xt_col * zs1_col, xt_col * zs2_col]
            xt_at_components = [xt_col * at_col]
            
                
            M = np.concatenate(
                [np.ones([n, 1])] + 
                [zs1_col, zs2_col, xt_col, at_col] + 
                xt_zs_components + 
                xt_at_components + 
                [zs1_col * at_col, zs2_col * at_col, urt.reshape(-1, 1)], 
                axis=1
            )
            
            rt = np.cbrt(M @ lmbda_scaled)
            return rt

    elif name == 'm4': # nonadditive with varying degree of deviation from single scalar noise
        def f_x0(zs, ux0, z_coef=1, v_coef=0, **kwargs):
            n = zs.shape[0]
            x0 = np.zeros((n, 2))

            # the first dimension
            gamma0_1 = np.array([-0.45, 1.5 * z_coef, 1, 0.5])
            M_1 = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                    ux0[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            x0_1 = np.cbrt(M_1 @ gamma0_1) + v_coef * ux0[:, 2]
            x0[:, 0] = x0_1

            # the second dimension
            gamma0_2 = np.array([-0.75, 2.5 * z_coef, 1, 1, 0.5])
            M_2 = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                    ux0[:, 1].reshape(-1, 1),
                    ux0[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            x0_2 = np.cbrt(M_2 @ gamma0_2) + v_coef * ux0[:, 2]
            x0[:, 1] = x0_2

            return x0
        
        def f_xt(zs, xtm1, atm1, uxt, z_coef=1, v_coef=0, **kwargs):
            xtm1_1 = xtm1[:, 0].reshape(-1, 1)
            xtm1_2 = xtm1[:, 1].reshape(-1, 1)
            n = xtm1.shape[0]
            xt = np.zeros((n, 2))

            # the first dimension
            gamma_1 = np.array([-0.45, 0.3, 0.3 * z_coef, 0.45 * z_coef, 0.5, 0.25]) 
            M_1 = np.concatenate(
                [
                    np.ones([n, 1]),
                    xtm1_1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_1 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                    uxt[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            xt_1 = np.cbrt(M_1 @ gamma_1) + v_coef * uxt[:, 2]
            xt[:, 0] = xt_1

            # the second dimension
            gamma_2 = np.array([-0.9, 0.4, 0.2, 0.6 * z_coef, 0.2 * z_coef, 0.9 * z_coef, 0.5, 0.9, 0.5])
            M_2 = np.concatenate(
                [
                    np.ones([n, 1]),
                    xtm1_1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_2 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_1 * (zs - 0.5),
                    xtm1_2 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                    uxt[:, 1].reshape(-1, 1),
                    uxt[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            xt_2 = np.cbrt(M_2 @ gamma_2) + v_coef * uxt[:, 2]
            xt[:, 1] = xt_2

            return xt

        def f_rt(zs, xt, at, urt, z_coef=1, v_coef=0, **kwargs):
            xt_1 = xt[:, 0].reshape(-1, 1)
            xt_2 = xt[:, 1].reshape(-1, 1)
            lmbda = np.array([-0.3, 0.2 * z_coef, 0.5, 0.2 * z_coef, 0.5, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt_1 * zs, xt_1 * at, xt_2 * zs, xt_2 * at, zs * at, urt], axis=1
            )
            rt = np.cbrt(M @ lmbda) + v_coef * urt.flatten()
            return rt

    elif name == 'm5': # additive with varying degree of deviation from single scalar noise
        def f_x0(zs, ux0, z_coef=1, v_coef=0, **kwargs):
            gamma0 = np.array([-0.3, 1 * z_coef, 1])
            n = zs.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                ],
                axis=1,
            )
            x0 = M @ gamma0
            x0 = x0.reshape(-1, 1)
            return x0

        def f_xt(zs, xtm1, atm1, uxt, z_coef=1, v_coef=0, **kwargs):
            gamma = np.array([-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1])
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    (zs - 0.5),
                    xtm1 * uxt[:, 1].reshape(-1, 1) * v_coef,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                ],
                axis=1,
            )
            xt = M @ gamma
            xt = xt.reshape(-1, 1)
            return xt

        def f_rt(zs, xt, at, urt, z_coef=1, v_coef=0, **kwargs):
            lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt * urt * v_coef, zs, at, xt * zs, xt * at, zs * at, urt], axis=1
            )
            rt = M @ lmbda
            return rt

    elif name == 'm6': # additive with varying degree of deviation from strict monotonicity
        def f_x0(zs, ux0, z_coef=1, v_coef=0, **kwargs):
            gamma0 = np.array([-0.3, 1 * z_coef, 1])
            n = zs.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                ],
                axis=1,
            )
            x0 = M @ gamma0
            x0 = x0.reshape(-1, 1)
            return x0

        def f_xt(zs, xtm1, atm1, uxt, z_coef=1, v_coef=0, **kwargs):
            gamma = np.array([-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1])
            n = xtm1.shape[0]
            M = np.concatenate(
                [
                    np.ones([n, 1]),
                    (zs - 0.5),
                    -np.abs(xtm1) * uxt[:, 0].reshape(-1, 1) * v_coef,
                    atm1.reshape(-1, 1) - 0.5,
                    xtm1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                ],
                axis=1,
            )
            xt = M @ gamma
            xt = xt.reshape(-1, 1)
            return xt

        def f_rt(zs, xt, at, urt, z_coef=1, v_coef=0, **kwargs):
            lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), -np.abs(xt) * urt * v_coef, zs, at, xt * zs, xt * at, zs * at, urt], axis=1
            )
            rt = M @ lmbda
            return rt

    elif name == 'm7': # nonadditive with varying degree of deviation from strict monotonicity
        def f_x0(zs, ux0, z_coef=1, v_coef=0, **kwargs):
            n = zs.shape[0]
            x0 = np.zeros((n, 2))

            # the first dimension
            gamma0_1 = np.array([-0.45, 1.5 * z_coef, 1, 0.5])
            M_1 = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                    ux0[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            x0_1 = np.cbrt(M_1 @ gamma0_1) - v_coef *  (ux0[:, 0] + 0.5 * ux0[:, 2])
            x0[:, 0] = x0_1

            # the second dimension
            gamma0_2 = np.array([-0.75, 2.5 * z_coef, 1, 1, 0.5])
            M_2 = np.concatenate(
                [
                    np.ones([n, 1]),
                    zs,
                    ux0[:, 0].reshape(-1, 1),
                    ux0[:, 1].reshape(-1, 1),
                    ux0[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            x0_2 = np.cbrt(M_2 @ gamma0_2) - v_coef * (ux0[:, 0] + ux0[:, 1] + 0.5 *  ux0[:, 2])
            x0[:, 1] = x0_2

            return x0
        
        def f_xt(zs, xtm1, atm1, uxt, z_coef=1, v_coef=0, **kwargs):
            xtm1_1 = xtm1[:, 0].reshape(-1, 1)
            xtm1_2 = xtm1[:, 1].reshape(-1, 1)
            n = xtm1.shape[0]
            xt = np.zeros((n, 2))

            # the first dimension
            gamma_1 = np.array([-0.45, 0.3, 0.3 * z_coef, 0.45 * z_coef, 0.5, 0.25]) 
            M_1 = np.concatenate(
                [
                    np.ones([n, 1]),
                    xtm1_1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_1 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                    uxt[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            xt_1 = np.cbrt(M_1 @ gamma_1) - v_coef * (0.5 * uxt[:, 0] + 0.25 * uxt[:, 2])
            xt[:, 0] = xt_1

            # the second dimension
            gamma_2 = np.array([-0.9, 0.4, 0.2, 0.6 * z_coef, 0.2 * z_coef, 0.9 * z_coef, 0.5, 0.9, 0.5])
            M_2 = np.concatenate(
                [
                    np.ones([n, 1]),
                    xtm1_1 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_2 * (atm1.reshape(-1, 1) - 0.5),
                    xtm1_1 * (zs - 0.5),
                    xtm1_2 * (zs - 0.5),
                    (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
                    uxt[:, 0].reshape(-1, 1),
                    uxt[:, 1].reshape(-1, 1),
                    uxt[:, 2].reshape(-1, 1),
                ],
                axis=1,
            )
            xt_2 = np.cbrt(M_2 @ gamma_2) - v_coef * (0.5 * uxt[:, 0] + 0.9 * uxt[:, 1] + 0.5 * uxt[:, 2])
            xt[:, 1] = xt_2

            return xt

        def f_rt(zs, xt, at, urt, z_coef=1, v_coef=0, **kwargs):
            xt_1 = xt[:, 0].reshape(-1, 1)
            xt_2 = xt[:, 1].reshape(-1, 1)
            lmbda = np.array([-0.3, 0.2 * z_coef, 0.5, 0.2 * z_coef, 0.5, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt_1 * zs, xt_1 * at, xt_2 * zs, xt_2 * at, zs * at, urt], axis=1
            )
            rt = np.cbrt(M @ lmbda) - v_coef * urt.flatten()
            return rt
        
    return f_x0, f_xt, f_rt
    