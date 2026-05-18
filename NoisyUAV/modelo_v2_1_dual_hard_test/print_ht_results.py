import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

df = pd.read_csv(r'NoisyUAV\modelo_v2_1_dual_hard_test\figures_ht\ht_golden_results.csv')
y_true = df['label_bin']; y_pred = df['pred_bin']
drones = df[df['label_bin']==1]
t5   = drones[drones['target']==5]['correct'].mean()
rest = drones[drones['target']!=5]['correct'].mean()
delta = t5 - rest

print('=== HARD TEST - RESULTADOS FINALES ===')
print(f'Recall:    {recall_score(y_true,y_pred):.4f}')
print(f'Precision: {precision_score(y_true,y_pred):.4f}')
print(f'F1-Score:  {f1_score(y_true,y_pred):.4f}')
print(f'Accuracy:  {accuracy_score(y_true,y_pred):.4f}')
print('')
print(f'Recall T5 Taranis (NUNCA VISTO): {t5:.4f}  (n={len(drones[drones["target"]==5])})')
print(f'Recall otros drones (vistos):    {rest:.4f}  (n={len(drones[drones["target"]!=5])})')
print(f'Delta (T5 - Resto): {delta:+.4f}')
if abs(delta) < 0.05:
    print('CONCLUSION: Generalizacion EXCELENTE')
elif delta > -0.10:
    print('CONCLUSION: Generalizacion BUENA')
else:
    print('CONCLUSION: Memorizacion parcial')

print('')
print('Recall por target:')
names = {0:'DJI',1:'FutabaT14',2:'FutabaT7',3:'Graupner',5:'Taranis',6:'Turnigy'}
for t in sorted(drones['target'].unique()):
    sub = drones[drones['target']==t]
    marker = ' <-- HELD-OUT (nunca visto)' if t==5 else ''
    print(f'  T{t} {names.get(t,"?")}: {sub["correct"].mean():.4f}  (n={len(sub)}){marker}')
