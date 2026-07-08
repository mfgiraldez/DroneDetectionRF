import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix

csv_baseline = r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream\golden_results_dualstream.csv"
csv_ht = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual_hard_test\figuras_ht\ht_golden_results.csv"

df_base = pd.read_csv(csv_baseline)
df_ht = pd.read_csv(csv_ht)

# Alineamos para que usen las mismas convenciones
for df in [df_base, df_ht]:
    if 'label_bin' in df.columns and 'label' not in df.columns:
        df['label'] = df['label_bin']
    if 'pred_bin' in df.columns and 'predicted' not in df.columns:
        df['predicted'] = df['pred_bin']
    if 'target_multiclass' not in df.columns and 'target' in df.columns:
        df['target_multiclass'] = df['target']
    
    df['correct'] = (df['label'] == df['predicted']).astype(int)

# Especificidad total (Recall en Ruido, label==0)
def get_metrics(df):
    y_true = df['label']
    y_pred = df['predicted']
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)
    rec = tp / (tp + fn)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0
    f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
    acc = accuracy_score(y_true, y_pred)
    
    # Recall per target
    drones = df[df['label']==1]
    rec_per_target = {}
    for t in sorted(drones['target_multiclass'].unique()):
        rec_per_target[t] = drones[drones['target_multiclass']==t]['correct'].mean()
        
    return spec, rec, prec, f1, acc, rec_per_target

spec_b, rec_b, prec_b, f1_b, acc_b, rpt_b = get_metrics(df_base)
spec_h, rec_h, prec_h, f1_h, acc_h, rpt_h = get_metrics(df_ht)

print("="*50)
print("COMPARATIVA BASELINE (V2.1) vs HARD-TEST (V2.1 sin T5)")
print("="*50)
print(f"{'Métrica':<15} | {'Baseline':<10} | {'Hard-Test':<10} | {'Delta':<10}")
print("-" * 50)
print(f"{'Especificidad':<15} | {spec_b:.4f}     | {spec_h:.4f}     | {spec_h-spec_b:+.4f}")
print(f"{'Recall Total':<15} | {rec_b:.4f}     | {rec_h:.4f}     | {rec_h-rec_b:+.4f}")
print(f"{'Precision':<15} | {prec_b:.4f}     | {prec_h:.4f}     | {prec_h-prec_b:+.4f}")
print(f"{'F1-Score':<15} | {f1_b:.4f}     | {f1_h:.4f}     | {f1_h-f1_b:+.4f}")
print(f"{'Accuracy':<15} | {acc_b:.4f}     | {acc_h:.4f}     | {acc_h-acc_b:+.4f}")

print("\nRECALL POR EMISOR (TARGET)")
print("-" * 50)
print(f"{'Target':<15} | {'Baseline':<10} | {'Hard-Test':<10} | {'Delta':<10}")
for t in rpt_b.keys():
    b_val = rpt_b[t]
    h_val = rpt_h.get(t, 0)
    print(f"{t:<15} | {b_val:.4f}     | {h_val:.4f}     | {h_val-b_val:+.4f}")
