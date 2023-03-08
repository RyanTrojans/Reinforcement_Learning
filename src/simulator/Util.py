def get_purity(state):
    return state[-2] / (state[-1] + state[-2])


def get_reward(purity, product,
               product_requirement,
               purity_requirement,
               failure_cost,
               product_price,
               yield_penalty_cost):
    if purity < purity_requirement:
        return - failure_cost

    if product < product_requirement:
        return product_price * product - yield_penalty_cost * (product_requirement - product)

    return product_price * product_requirement


def select_action(chrom_dat):
    raw = chrom_dat[['action', 'q_p', 'q_i']].values
    selected_windows = []
    for a, i, j in raw:
        selected = True
        for u, m, n in raw:
            if a != u and i <= m and j >= n:
                selected = False
        if selected:
            selected_windows.append([i, j])
    return selected_windows
