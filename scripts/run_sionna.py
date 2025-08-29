import argparse

# === Original code (wrapped) ===
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# === Helper functions ===

def train_quantile_models(X_train, Y_train, alpha=0.1):
    lower = GradientBoostingRegressor(loss='quantile', alpha=alpha/2).fit(X_train, Y_train)
    upper = GradientBoostingRegressor(loss='quantile', alpha=1 - alpha/2).fit(X_train, Y_train)
    return lower, upper
def get_features_and_treatment(df):
    X = df[['x_pos', 'y_pos', 'z']].values
    T = df['T'].values.reshape(-1, 1)
    return np.hstack([X, T])

# === Eta Calibration (Lei & Candès with weights) ===
def calibrate_eta(df_cal, q_lo, q_hi, thresh, alpha=0.2, delta=0.1, max_eta=10, n_eta=100):
    etas = np.linspace(0, max_eta, n_eta)
    features = ['x_pos', 'y_pos', 'z']
    df_macro = df_cal[df_cal['T'] == 0].reset_index(drop=True)

    n = len(df_macro)
    for eta in etas:
        weighted_losses = []
        for _, row in df_macro.iterrows():
            x = row[features].values.reshape(1, -1)
            y = row['Y']

            dist = np.linalg.norm(row[['x_pos', 'y_pos']].values - np.array([8.5, 21]))
            e = 1 if dist > thresh else 0
            e = max(e, 1e-6)
            w = 1 / e

            xt = np.hstack([x, [[1]]])  # t=1 (counterfactual for pico)
            ql = q_lo.predict(xt)[0]
            qh = q_hi.predict(xt)[0]

            loss = 0 if ql - eta <= y <= qh + eta else 1
            weighted_losses.append(w * loss)

        L_hat = np.mean(weighted_losses)
        bound = L_hat + np.sqrt(np.log(1 / delta) / (2 * n))
        if bound <= alpha:
            return eta
    return max_eta

# === Eta Calibration (Augmented) ===
def calibrate_eta_augmented(df_cal, q_lo, q_hi, cf_model, thresh, alpha=0.2, delta=0.1, max_eta=10, n_eta=100, p_treated=0.2):
    etas = np.linspace(0, max_eta, n_eta)
    p_control = 1 - p_treated

    features = ['x_pos', 'y_pos', 'z']
    df_real = df_cal[df_cal['T'] == 1].reset_index(drop=True)
    df_synth = df_cal[df_cal['T'] == 0].reset_index(drop=True)

    n_real = len(df_real)
    n_synth = len(df_synth)
    r = n_synth // max(1, n_real)

    for eta in etas:
        losses = []
        for i, real_row in df_real.iterrows():
            x_real = real_row[features].values.reshape(1, -1)
            y_real = real_row['Y']

            dist_real = np.linalg.norm(real_row[['x_pos', 'y_pos']].values)
            e_real = 1 if dist_real > thresh else 0
            e_real = max(e_real, 1e-6)
            w_real = p_treated / e_real

            xt = np.hstack([x_real, [[1]]])  # t=1
            ql_real = q_lo.predict(xt)[0]
            qh_real = q_hi.predict(xt)[0]

            loss_obs = 0 if ql_real - eta <= y_real <= qh_real + eta else 1
            y_plugin = max(0.0, cf_model.predict(xt)[0])
            loss_plugin = 0 if ql_real - eta <= y_plugin <= qh_real + eta else 1
            correction = w_real * (loss_obs - loss_plugin)

            # synthetic points for this real point
            start_idx = i * r
            end_idx = min((i + 1) * r, n_synth)
            synth_contrib = 0

            for j in range(start_idx, end_idx):
                synth_row = df_synth.iloc[j]
                x_synth = synth_row[features].values.reshape(1, -1)
                y_synth = synth_row['Y']

                dist_synth = np.linalg.norm(synth_row[['x_pos', 'y_pos']].values)
                e_synth = 1 if dist_synth > thresh else 0
                e_synth = min(1 - 1e-6, max(e_synth, 1e-6))
                w_synth = p_control / (1 - e_synth)

                xt_s = np.hstack([x_synth, [[1]]])
                ql_synth = q_lo.predict(xt_s)[0]
                qh_synth = q_hi.predict(xt_s)[0]

                loss_synth = 0 if ql_synth - eta <= y_synth <= qh_synth + eta else 1
                synth_contrib += w_synth * loss_synth

            avg_synth_loss = synth_contrib / max(1, (end_idx - start_idx))
            total_loss = avg_synth_loss + correction
            losses.append(total_loss)

        L_hat = np.mean(losses)
        bound = L_hat + np.sqrt(np.log(1 / delta) / (2 * n_real))
        if bound <= alpha:
            return eta
    return max_eta

# === Interval Computation ===
def get_interval(x, t, q_lo, q_hi, eta):
    x_t = np.hstack([x, t]).reshape(1, -1)
    lo = q_lo.predict(x_t)[0] - eta
    hi = q_hi.predict(x_t)[0] + eta
    return lo, hi

# === Counterfactual Estimator ===
def train_counterfactual_model(df):
    X = get_features_and_treatment(df)
    Y = df['Y'].values
    model = GradientBoostingRegressor()
    model.fit(X, Y)
    return model



# === Parameters ===
n_trials = 50  # Number of random trials
fixed_thresh = 80  # Fixed threshold for the policy
alpha = 0.1  # Target miscoverage level
delta = 0.1  # Confidence level for bound

# === Storage ===
eta_lc_list = []
eta_aug_list = []

# === Load Data ===
data = np.load("user_connection_data.npz")
X = data["user_positions"]
Y_macro = data["rss_macro"]
Y_pico = data["rss_pico"]
T = np.array([0 if c == "macro" else 1 for c in data["connected_to"]])
Y_obs = np.where(T == 0, Y_macro, Y_pico)

df_all = pd.DataFrame(X, columns=["x_pos", "y_pos", "z"])
df_all["T"] = T
df_all["Y"] = Y_obs
df_all["Y_macro"] = Y_macro
df_all["Y_pico"] = Y_pico

# === Storage ===
avg_width_lc = []
avg_width_aug = []

for seed in tqdm(range(n_trials), desc="Running trials"):
    df = df_all.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_qtrain = df.iloc[:100]  # quantile regressor
    df_cf = df.iloc[100:500]  # counterfactual model
    df_qcal = df.iloc[500:1000]  # calibration set

    # === Train quantile models ===
    X_q = get_features_and_treatment(df_qtrain)
    y_q = df_qtrain["Y"].values
    q_lo, q_hi = train_quantile_models(X_q, y_q, alpha=alpha)

    # === Train counterfactual model ===
    cf_model = train_counterfactual_model(df_cf)


    # === Estimate p_treated for fixed threshold ===
    def policy_fn(x):
        dist = np.linalg.norm(x[:2] - np.array([8.5, 21]))
        return 1 if dist >= fixed_thresh else 0


    p_treated = np.mean([policy_fn(row[['x_pos', 'y_pos', 'z']].values) for _, row in df_qcal.iterrows()])

    # === Calibrate etas ===
    eta_lc = calibrate_eta(df_qcal, q_lo, q_hi, fixed_thresh, alpha=alpha, delta=delta)
    eta_aug = calibrate_eta_augmented(df_qcal, q_lo, q_hi, cf_model, fixed_thresh, alpha=alpha, delta=delta,
                                      p_treated=p_treated)

    # === Compute average width over calibration set ===
    X_cal = get_features_and_treatment(df_qcal)
    qlo_vals = q_lo.predict(X_cal)
    qhi_vals = q_hi.predict(X_cal)

    width_lc = np.mean(qhi_vals - qlo_vals + 2 * eta_lc)
    width_aug = np.mean(qhi_vals - qlo_vals + 2 * eta_aug)

    avg_width_lc.append(width_lc/10)
    avg_width_aug.append(width_aug/10)

# === Plot Width Distributions ===
# === Configure plot font ===
matplotlib.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'font.family': 'serif'
})
plt.figure(figsize=(10, 5))
plt.hist(avg_width_lc, bins=20, alpha=0.7, label='Average width CCI')
plt.hist(avg_width_aug, bins=20, alpha=0.7, label='Average width SP-CCI')
plt.axvline(np.mean(avg_width_lc), color='blue', linestyle='dashed', label=f'Mean CCI: {np.mean(avg_width_lc):.2f}')
plt.axvline(np.mean(avg_width_aug), color='orange', linestyle='dashed', label=f'Mean SP-CCI: {np.mean(avg_width_aug):.2f}')
plt.xlabel("Average Interval Width")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_width_sionna.pdf")
plt.show()


def main():
    parser = argparse.ArgumentParser(description="Sionna experiment runner")
    parser.add_argument("--out", type=str, default="results/sionna.json", help="Path to JSON results")
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
    print("Please adapt scripts/Sionna.py to call your experiment entrypoint.")

if __name__ == "__main__":
    main()