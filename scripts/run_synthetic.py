import argparse

# === Original code (wrapped) ===
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from econml.metalearners import XLearner
import seaborn as sns
from scipy.stats import beta
from scipy.special import expit

# Step 1: Simulate data according to Lei & Candes Section 3.6

def simulate_lei_candes(n=5000, d=25, rho=0.0, hetero=False, seed=42):
    np.random.seed(seed)
    # Equicorrelated Gaussian
    cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    Z = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    X = expit(Z)

    def f(x):
        return 2 / (1 + np.exp(-12 * (x - 0.5)))

    mu1 = f(X[:, 0]) * f(X[:, 1])
    sigma = np.sqrt(-np.log(X[:, 0])) if hetero else 1.0
    Y1 = mu1 + sigma * np.random.normal(size=n)
    Y0 = np.zeros(n)
    e = 0.4*beta.cdf(X[:, 0], a=2, b=4)
    T = np.random.binomial(1, e)
    Y = T * Y1 + (1 - T) * Y0

    return pd.DataFrame({**{f'x{i}': X[:, i] for i in range(d)},
                         'T': T,
                         'Y': Y,
                         'Y0': Y0,
                         'Y1': Y1,
                         'e': e})

def split_data(df):
    rand = np.random.randint(1e6)
    df_train, df_temp = train_test_split(df, test_size=0.6, stratify=df['T'], random_state=rand)
    df_cal, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['T'], random_state=rand + 1)
    return df_train, df_cal, df_test

def train_quantile_models(X_train, Y_train, alpha=0.1):
    lower = GradientBoostingRegressor(loss='quantile', alpha=alpha/2).fit(X_train, Y_train)
    upper = GradientBoostingRegressor(loss='quantile', alpha=1 - alpha/2).fit(X_train, Y_train)
    return lower, upper

def train_counterfactual_model(df_train, features):
    X = df_train[features].values
    y = df_train['Y'].values
    t = df_train['T'].values
    x_learner = XLearner(models=GradientBoostingRegressor(), propensity_model=LogisticRegression())
    x_learner.fit(y, t, X=X)
    return x_learner

def generate_synthetic_Y1(model, data, features):
    X = data[features].values
    tau = model.effect(X)
    return np.zeros(len(X)) + tau  # since Y0 = 0

from scipy.stats import beta
import numpy as np

def calibrate_eta(df_cal, q_lo, q_hi, cf_model, alpha=0.1, delta=0.1, max_eta=10, n_eta=50, p_treated=0.5):
    etas = np.linspace(0, max_eta, n_eta)
    p_control = 1 - p_treated

    df_real = df_cal[df_cal['T'] == 1].reset_index(drop=True)
    df_synth = df_cal[df_cal['T'] == 0].reset_index(drop=True)
    n_real = len(df_real)
    n_synth = len(df_synth)
    r = n_synth // n_real

    for eta in etas:
        losses = []

        for i, (_, real_row) in enumerate(df_real.iterrows()):
            total_weight = 0
            x_real = real_row[[f'x{i}' for i in range(25)]].values.reshape(1, -1)
            y_real = real_row['Y']
            x1_real = real_row['x0']
            e_real = 0.4*beta.cdf(x1_real, a=2, b=4)
            w_real = p_treated / e_real
            total_weight += w_real

            ql_real = q_lo.predict(x_real)[0]
            qh_real = q_hi.predict(x_real)[0]

            loss_obs = 0 if ql_real - eta <= y_real <= qh_real + eta else 1
            y_plugin = cf_model.effect(x_real)[0]
            loss_plugin = 0 if ql_real - eta <= y_plugin <= qh_real + eta else 1
            correction = w_real * (loss_obs - loss_plugin)

            # Synthetic samples assigned to this real point
            start_idx = i * r
            end_idx = min((i + 1) * r, n_synth)
            synth_contrib = 0

            for j in range(start_idx, end_idx):
                synth_row = df_synth.iloc[j]
                x_synth = synth_row[[f'x{i}' for i in range(25)]].values.reshape(1, -1)
                y_synth = synth_row['Y']
                x1_synth = synth_row['x0']
                e_synth = 0.4*beta.cdf(x1_synth, a=2, b=4)
                w_synth = p_control / (1 - e_synth)
                ql_synth = q_lo.predict(x_synth)[0]
                qh_synth = q_hi.predict(x_synth)[0]
                loss_synth = 0 if ql_synth - eta <= y_synth <= qh_synth + eta else 1
                synth_contrib += w_synth * loss_synth
                total_weight += w_synth / max(1, (end_idx - start_idx))

            avg_synth_loss = synth_contrib / max(1, (end_idx - start_idx))
            total_loss = avg_synth_loss + correction
            losses.append(total_loss)

        L_hat = np.mean(losses)
        bound = L_hat + np.sqrt(np.log(1 / delta) / (2 * n_real))
        if bound <= alpha:
            return eta

    return max_eta


def calibrate_eta_real(df_cal, q_lo, q_hi, cf_model, alpha=0.1, delta=0.1, max_eta=10, n_eta=50, p_treated=0.5):
    etas = np.linspace(0, max_eta, n_eta)
    df_real = df_cal[df_cal['T'] == 1].reset_index(drop=True)
    df_synth = df_cal[df_cal['T'] == 0].reset_index(drop=True)
    n_real = len(df_real)
    for eta in etas:
        losses = []
        for i, (_, real_row) in enumerate(df_real.iterrows()):
            total_weight = 0
            x_real = real_row[[f'x{i}' for i in range(25)]].values.reshape(1, -1)
            y_real = real_row['Y']
            x1_real = real_row['x0']
            e_real = 0.4*beta.cdf(x1_real, a=2, b=4)
            w_real = p_treated / e_real
            total_weight += w_real
            ql_real = q_lo.predict(x_real)[0]
            qh_real = q_hi.predict(x_real)[0]
            loss_obs = 0 if ql_real - eta <= y_real <= qh_real + eta else 1
            correction = w_real * (loss_obs)
            total_loss = (correction) / total_weight
            losses.append(total_loss)
        L_hat = np.mean(losses)
        bound = L_hat + np.sqrt(np.log(1 / delta) / (2 * n_real))
        if bound <= alpha:
            return eta
    return max_eta



def evaluate_coverage_and_eta(df, features, alpha=0.1, delta=0.1, train_frac=1.0):
    df_train, df_cal, df_test = split_data(df)
    df_train = df_train.sample(frac=train_frac, random_state=42)
    treated_train = df_train[df_train['T'] == 1]
    X_treated = treated_train[features].values
    Y_treated = treated_train['Y'].values
    q_lo, q_hi = train_quantile_models(X_treated, Y_treated, alpha)
    p_treated = (df_train['T'] == 1).mean()

    cf_model = train_counterfactual_model(df_train, features)

    control_cal = df_cal[df_cal['T'] == 0].copy()
    control_cal['Y'] = generate_synthetic_Y1(cf_model, control_cal, features)
    real_cal = df_cal[df_cal['T'] == 1][features + ['Y', 'T']]
    aug_cal = pd.concat([real_cal, control_cal[features + ['Y', 'T']]], ignore_index=True)

    eta_real = calibrate_eta_real(real_cal, q_lo, q_hi, cf_model, alpha, delta, p_treated= p_treated)
    eta_aug = calibrate_eta(aug_cal, q_lo, q_hi,cf_model,  alpha, delta, p_treated= p_treated)

    tested = df_test[df_test['T'] == 0]
    X_test = tested[features].values
    Y1_true = tested['Y1'].values
    ql = q_lo.predict(X_test)
    qh = q_hi.predict(X_test)

    intervals_real = np.stack([ql - eta_real, qh + eta_real], axis=1)
    intervals_aug = np.stack([ql - eta_aug, qh + eta_aug], axis=1)

    cov_real = np.mean((Y1_true >= intervals_real[:, 0]) & (Y1_true <= intervals_real[:, 1]))
    cov_aug = np.mean((Y1_true >= intervals_aug[:, 0]) & (Y1_true <= intervals_aug[:, 1]))

    width_aug = np.mean((qh - ql) + 2 * eta_aug)
    width_real = np.mean((qh - ql) + 2 * eta_real)

    return cov_real, cov_aug, eta_real, eta_aug, width_real, width_aug

def run_multiple_simulations(n_runs=5, alpha=0.1, delta=0.1):
    df = simulate_lei_candes()
    features = [col for col in df.columns if col.startswith('x')]

    quality_levels = {
        'SP-CCI (LQ CF labels)': 0.2,
        'SP-CCI (MQ CF labels)': 0.6,
        'SP-CCI (HQ CF labels)': 1.0
    }

    results = {
        'CCI': {'coverage': [], 'eta': [], 'width': []}
    }
    for q in quality_levels:
        results[q] = {'coverage': [], 'eta': [], 'width': []}

    for _ in range(n_runs):
        for label, frac in quality_levels.items():
            cov_r, cov_a, eta_r, eta_a, width_r, width_a = evaluate_coverage_and_eta(df, features, alpha, delta, train_frac=frac)
            if label == 'SP-CCI (LQ CF labels)':
                results['CCI']['coverage'].append(cov_r)
                results['CCI']['eta'].append(eta_r)
                results['CCI']['width'].append(width_r)
            results[label]['coverage'].append(cov_a)
            results[label]['eta'].append(eta_a)
            results[label]['width'].append(width_a)

    mpl.rcParams.update({
        "font.size": 18,                # Default font size
        "axes.titlesize": 18,           # Title font size
        "axes.labelsize": 18,           # Axis label font size
        "xtick.labelsize": 18,          # X-axis tick font size
        "ytick.labelsize": 18,          # Y-axis tick font size
        "legend.fontsize": 16,          # Legend font size
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False
    })

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    label_colors = {
        "CCI": '#4c72b0',
        "SP-CCI (LQ CF labels)": "orange",
        "SP-CCI (MQ CF labels)": "green",
        "SP-CCI (HQ CF labels)": "red"
    }

    plt.rcParams["axes.formatter.use_mathtext"] = True  # Optional: suppress font warning

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for label in results:
        data = np.array(results[label]['coverage'])
        if np.std(data) < 1e-8:
            data += np.random.normal(0, 1e-6, size=len(data))

        color = label_colors[label]
        sns.kdeplot(data, label=label, fill=True, ax=ax, color=color)

        # Plot mean as dashed line in same color
        mean_val = np.mean(data)
        ax.axvline(mean_val, color=color, linestyle='--')

    # Target coverage line
    ax.axvline(1 - alpha, color='black', linestyle='--', label='Target Coverage')
    ax.set_xlim(right=1)
    ax.set_xlabel('Empirical Coverage')
    ax.set_ylabel('Density')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("Coverage.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    for label in results:
        data = np.array(results[label]['eta'])
        if np.std(data) < 1e-8:
            data += np.random.normal(0, 1e-6, size=len(data))
        sns.kdeplot(data, label=label, fill=True)
    plt.xlabel(r'Widening Parameter $\eta$')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig("width.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    for label in results:
        data = np.array(results[label]['width'])
        if np.all(np.isnan(data)):
            continue
        data = data[~np.isnan(data)]
        if np.std(data) < 1e-8:
            data += np.random.normal(0, 1e-6, size=len(data))
        sns.kdeplot(data, label=label, fill=True)
    plt.xlabel('Average Prediction Interval Width')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_width.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    threshold = 0.85
    print("\nPercentage of runs with coverage < {:.2f}:".format(threshold))
    for label in results:
        coverages = np.array(results[label]["coverage"])
        below_thresh = np.mean(coverages < threshold) * 100
        print(f"{label}: {below_thresh:.1f}% of runs")

run_multiple_simulations(n_runs=50, alpha=0.15, delta=0.1)


def main():
    parser = argparse.ArgumentParser(description="Synthetic experiment runner")
    parser.add_argument("--out", type=str, default="results/synth.json", help="Path to JSON results")
    args = parser.parse_args()
    # You may need to call a function from the original code that triggers a run
    # Here we try common function names:
    for fn_name in ["run", "main", "experiment", "train"]:
        fn = globals().get(fn_name, None)
        if callable(fn):
            try:
                return fn(args)
            except TypeError:
                try:
                    return fn()
                except Exception:
                    pass
    print("Please adapt scripts/Synthetic.py to call your experiment entrypoint.")

if __name__ == "__main__":
    main()