import numpy as np

def get_env(name):
    if name == 'm1': # nonadditive
        def f_x0(zs, ux0, z_coef=1):
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
        
        def f_xt(zs, xtm1, atm1, uxt, z_coef=1):
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

        def f_rt(zs, xt, at, urt, z_coef=1):
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
        def f_x0(zs, ux0, z_coef=1):
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

        def f_xt(zs, xtm1, atm1, uxt, z_coef=1):
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

        def f_rt(zs, xt, at, urt, z_coef=1):
            lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef, 1])
            n = xt.shape[0]
            at = at.reshape(-1, 1)
            M = np.concatenate(
                [np.ones([n, 1]), xt, zs, at, xt * zs, xt * at, zs * at, urt], axis=1
            )
            rt = M @ lmbda
            return rt
        
    return f_x0, f_xt, f_rt
    