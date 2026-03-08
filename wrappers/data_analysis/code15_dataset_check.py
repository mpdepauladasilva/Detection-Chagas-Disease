import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/datasets_original_csv/code15_chagas_labels.csv')

# Agrupar por patient_id e contar exames e status de chagas
resultado = df.groupby('patient_id').agg(
    total_exames=('exam_id', 'count'),
    chagas_true=('chagas', lambda x: (x == True).sum()),
    chagas_false=('chagas', lambda x: (x == False).sum())
).reset_index()

# Mostrar o resultado
print(resultado)

# Se quiser salvar em um novo arquivo CSV
resultado.to_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/datasets_original_csv/resultado_pacientes.csv', index=False)