import jax.numpy as jnp
import pandas as pd

def suggest_eta_max(q, scale, k=5.0):
    """Upper bound on |eta| to avoid softmax saturation."""
    return float(k) / (float(scale) * jnp.sqrt(float(q)))

q_list = [10, 20, 50, 100]
scale_a = 1.0     # eta_a
scale_b = 1.0    # eta_b

records = []
for q in q_list:
    eta_a_max = suggest_eta_max(q, scale_a)
    eta_b_max = suggest_eta_max(q, scale_b)

    records.append({
        "q": q,
        "scale(eta_a)": scale_a,
        "η_a max": eta_a_max,
        "recommended η_a range": f"[{-eta_a_max:.3f}, {eta_a_max:.3f}]",
        "scale(eta_b)": scale_b,
        "η_b max": eta_b_max,
        "recommended η_b range": f"[{-eta_b_max:.3f}, {eta_b_max:.3f}]",
    })

df = pd.DataFrame(records)
print(df.to_string(index=False))
