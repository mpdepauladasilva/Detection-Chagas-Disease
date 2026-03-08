import pandas as pd

# Carregar os arquivos
df1 = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv')  # Seu primeiro CSV com coluna 'record_id'
df2 = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/training_data_indices.csv')    # Segundo CSV com coluna 'record'

# Extrair números dos records do segundo arquivo
df2['id'] = df2['record'].str.extract(r'/(\d+)$')

# Verificar presença
df2['presente'] = df2['id'].isin(df1['record_id'].astype(str))

# Salvar resultado
df2.to_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/final_compare.csv', index=False)

print("Comparação concluída! Arquivo salvo como 'comparacao_final.csv'")