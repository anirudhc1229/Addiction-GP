import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

goals = []
hits = []
weights = []
st_cut = 0 # hits including and after this index are counted in short term

st_diffs = []
lt_diffs = []

INIT_DROP = 0.9
MAX_ST_ERR = 10

def f_lin(a, t):
    return a[0] + a[1] * t

def f_sinu(b, t):
    return b[0] + b[1] * t + b[2] * np.sin(b[3] * t + b[4])

def err_sinu(b, t, y):
    return f_sinu(b, t) - y

def f_quad(c, t):
    # return c[0] + c[1] * t + c[2] * t**2
    return c[0] + c[1] * (t - c[2])**2

def w_err_quad(c, t, y, w):
    return np.multiply(w, (f_quad(c, t) - y))

# plot fun (t -> y) w/ params x
def plot(fun, x, t, y):
    print(len(t), len(y))
    print(t, y)
    plt.scatter(t, y, c='r')
    t_plt = np.linspace(0, len(t) - 1, 1000)
    plt.plot(t_plt, fun(x, t_plt))
    plt.show()

# def count_extrema():
#     count = 0
#     for i in range(1, len(goals) - 1):
#         count += goals[i] < goals[i - 1] and goals[i] < goals[i + 1]
#         count += goals[i] > goals[i - 1] and goals[i] > goals[i + 1]
#     return count

# linear lsq on latest data
def short_term():
    global st_cut
    days = np.arange(0, len(hits[st_cut:]))
    A = np.vstack([np.ones(len(days)), days]).T
    sol = scipy.optimize.lsq_linear(A, hits[st_cut:])
    if len(hits[st_cut:]) > 2:
        print(f"ERR: {sol.cost}")
        if sol.cost > MAX_ST_ERR:
            st_cut = len(hits) - 2 # reset to only last 2 points
            return short_term()
    print(sol)
    plot(f_lin, sol.x, days, hits[st_cut:])
    est = f_lin(sol.x, len(hits[st_cut:]))
    return est.item()

# sinusoidal lsq on all data
def long_term():
    days = np.arange(0, len(hits))
    b_init = [hits[0], -1, 1, 1, 0]
    sol = scipy.optimize.least_squares(
        err_sinu, b_init, args=(days, hits))
    est = f_sinu(sol.x, len(hits))
    print(sol)
    plot(f_sinu, sol.x, days, hits)
    return est

# # quadratic lsq on diff vs. w_st
# def weigh():
#     print(f'goals: {goals} hits: {hits}')
#     diff = np.absolute(np.asarray(goals) - np.asarray(hits[1:]))
#     w = np.arange(len(weights)) + 1 # [1, n + 1]
#     bounds = scipy.optimize.Bounds([-np.inf, np.inf], 
#                                    [0, np.inf], # only concave up
#                                    [0, 1]) # sol between 0 and 1
#     lb = [-np.inf, 0, 0]
#     ub = [np.inf, np.inf, 1]
#     c_init = [0, 1, 0.5]
#     sol = scipy.optimize.least_squares(
#         w_err_quad, c_init, bounds=(lb, ub), args=(weights, diff, w))
#     print(sol)
#     plot(f_quad, sol.x, weights, diff)
#     w_st = sol.x[2] # min of parabola
#     weights.append(w_st)
#     print(f"w_st {w_st}")
#     return np.array([w_st, 1 - w_st]) # complementary

def weigh():
    bias = np.arange(len(weights)) + 1 # linear weighting
    s = np.sum(np.multiply(st_diffs, bias))
    l = np.sum(np.multiply(lt_diffs, bias))
    return np.array([l / (s + l), s / (s + l)]) # (st, lt)

# weighted sum of st and lt
def combined_goal(st, lt):
    g = np.array([st, lt])
    w = weigh()
    print(f"WEIGHTS (st, lt): {w}")
    est = w.dot(g) # weighted sum
    print(f"ST: {st}\tLT: {lt}\tEST: {est}")
    return min(hits[-1], max(0, math.floor(est)))

def get_goal():
    goal = 0
    if len(hits) == 1:  
        goal = math.floor(INIT_DROP * hits[0])
        weights.append(0.5)
    elif hits[-1] == 0:
        goal = 0
        weights.append(0.5)
    else:
        st = short_term()
        st_diffs.append(abs(st - hits[-1]))
        lt = long_term()
        lt_diffs.append(abs(lt - hits[-1]))
        goal = combined_goal(st, lt)
    goals.append(goal)
    return goal

def add_hit(hit):
    hits.append(hit)

def run_day(hit: int) -> int:
    hits.append(hit)
    goal = get_goal()
    return goal

# if __name__ == '__main__':
#     while True:
#         hit = int(input("add hit: "))
#         hits.append(hit)
#         goal = get_goal()
#         print(f"goal: {goal}")
#         # if hits[-1] == 0:
#         #     print("Quit successfully!")
#         #     break