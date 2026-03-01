import pandas as pd

# Configurações
arquivo_original = '/media/mpsilva-lx/Dados/Recursos/arquivos_nao_encontrados.csv'
#arquivo_novo = '/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv'
arquivo_novo = '/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/arquivos_nao_encontrados_absolut.csv'

# Ler o arquivo e modificar os paths em uma única operação
df = pd.read_csv(arquivo_original)

# Modificar ambas as colunas
for coluna in ['file_path', 'dat_path']:
    df[coluna] = df[coluna].str.extract(r'(/Dataset-Chagas/.*)', expand=False)

# Salvar o resultado
df.to_csv(arquivo_novo, index=False)
print(f"Arquivo salvo com sucesso: {arquivo_novo}")
print(f"Total de registros processados: {len(df)}")