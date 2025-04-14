import glob, os, shutil
import pandas as pd
import accounting
from root import FileSys
from accounting import NFSe, NFSeColl
import time
import PyPDF2
from datetime import datetime
import argparse

def last_month_year():
    today = datetime.today()
    last_month = today.month - 1 if today.month > 1 else 12
    last_year = today.year if today.month > 1 else today.year - 1
    return last_year

def run_daily_balance(series_folder, year):
    lst_csvs = glob.glob(f"{series_folder}/{year}/{year}*/extrato_cc_*.csv")
    lst_dfs = []
    for f in lst_csvs:
        #print(f)
        _df = pd.read_csv(f, sep=";", dtype={"Valor": float}, parse_dates=["Data"])
        _df = _df[["Data", "Categoria", "Valor"]].dropna().copy()
        lst_dfs.append(_df.copy())

    df_full = pd.concat(lst_dfs)
    df_full.to_csv(f"{series_folder}/{year}/caixa-diario_cc_{year}.csv", sep=";", index=False)
    return df_full


def run_monthly_balance(df, ano_especifico, series_folder, year):
    """
    Agrega o DataFrame por totais mensais e tipos de lançamento,
    garantindo que meses sem lançamentos apareçam com valor zero para um ano específico,
    com as categorias agregadas em colunas e incluindo uma linha com os totais anuais.

    Parâmetros:
    df : DataFrame
        O DataFrame contendo as colunas 'data', 'Categoria' e 'Valor'.
    ano_especifico : int
        O ano para o qual se deseja garantir todos os meses com totais.

    Retorna:
    DataFrame
        O DataFrame com totais mensais por tipo de lançamento, com categorias como colunas,
        incluindo uma linha de totais anuais.
    """

    # Extrai o mês e o ano de cada data para agregação mensal
    df['Mes'] = df['Data'].dt.to_period('M')

    # Agrega o DataFrame por 'Mes' e 'Categoria', somando os valores
    result = df.groupby(['Mes', 'Categoria'])['Valor'].sum().reset_index()

    # Cria uma lista de todos os meses do ano específico
    meses = pd.period_range(f'{ano_especifico}-01', f'{ano_especifico}-12', freq='M')

    # Cria um DataFrame com todos os meses e tipos de lançamento possíveis
    tipos_lancamento = df['Categoria'].unique()
    meses_df = pd.MultiIndex.from_product([meses, tipos_lancamento], names=['Mes', 'Categoria'])

    # Cria um DataFrame vazio para os meses e tipos de lançamento
    result_full = pd.DataFrame(index=meses_df).reset_index()

    # Junta os dados agregados com os meses completos
    final_result = pd.merge(result_full, result, on=['Mes', 'Categoria'], how='left').fillna({'Valor': 0})

    # Agora, pivota o DataFrame para ter as categorias como colunas
    final_result_pivot = final_result.pivot(index='Mes', columns='Categoria', values='Valor').reset_index()

    # Preenche qualquer valor ausente com zero
    final_result_pivot = final_result_pivot.fillna(0)

    fpivot = final_result_pivot[['Mes', 'Receitas', 'Impostos', 'Prolabore', 'Custeio', 'Lucros']].copy()

    df_o = pd.DataFrame(
        {
            "Mes": fpivot["Mes"],
            "Receitas": fpivot["Receitas"],
            "Impostos": fpivot["Impostos"],
            "Custeio": fpivot["Custeio"],
            "Prolabore": fpivot["Prolabore"],
            "Lucros": fpivot["Lucros"],
            "Desembolsos": fpivot["Impostos"] + fpivot["Custeio"] + fpivot["Prolabore"] + fpivot["Lucros"],
        }
    )

    df_o["Saldo"] = df_o["Receitas"] + df_o["Desembolsos"]
    df_o["Saldo Acum"] = df_o["Saldo"].cumsum()

    df_o["Receitas Acum"] = df_o["Receitas"].cumsum()

    df_o["Desembolsos Acum"] = df_o["Desembolsos"].cumsum()

    df_o["Data"] = [str(s) + "-01" for s in df_o["Mes"]]
    df_o["Data"] = pd.to_datetime(df_o["Data"])

    lst_cols = [
        "Data",
        "Receitas", "Desembolsos", "Saldo",
        "Impostos", "Custeio", "Prolabore", "Lucros",
        "Receitas Acum", "Desembolsos Acum", "Saldo Acum",
    ]
    df_o = df_o[lst_cols].copy()
    # Convertendo para o último dia do mês
    df_o['Data'] = df_o['Data'] + pd.offsets.MonthEnd(0)

    # exportar
    df_o.to_csv(f"{series_folder}/{year}/caixa-mensal_cc_{year}.csv", sep=";", index=False)

    lst_cols = [
                "Saldo",
                "Receitas",
                "Desembolsos",
                "Impostos",
                "Custeio",
                "Prolabore",
                "Lucros",
            ]
    df_m = pd.DataFrame(
        {
            "Categoria": lst_cols,
            "Total": [df_o[field].sum() for field in lst_cols],
            "Média": [df_o[field].mean().round(2) for field in lst_cols]
        }
    )
    # exportar
    df_m.to_csv(f"{series_folder}/{year}/caixa-mensal-resumo_cc_{year}.csv", sep=";", index=False)

    # plotar relatório
    accounting.plot_yearly_cashflow(
        df=df_o,
        folder=f"{series_folder}/{year}",
        filename=f"caixa-mensal_cc_{year}",
    )

    return df_o, df_m


def run_declared_revenue(src_root, series_folder, year):
    # 1) Get all NFSe from projects
    # get all NFSe's from LOS ALAMOS
    lst_files = glob.glob(f"{src_root}/losalamos/*/*/budget/*.xml")
    if len(lst_files) == 0:
        pass
    else:
        nfe_col = NFSeColl()
        nfe_col.load_files(lst_files=lst_files)
        # Convert the 'date_column' to datetime
        nfe_col.catalog["Date"] = pd.to_datetime(nfe_col.catalog["Date"])
        df_nfe_full = nfe_col.catalog.copy()
        # filter to current year
        df_nfe_cur = df_nfe_full.query(f"Date >= '{year}-01-01' and Date < '{int(year)+1}-01-01'").copy()
        # Obter a segunda pasta acima para cada caminho
        df_nfe_cur["Projeto"] = [
            os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            for file_path in list(df_nfe_cur["File_Data"])
        ]
        # obter pdfs
        df_nfe_cur["File_Data_PDF"] = [
            os.path.splitext(file_path)[0] + ".pdf"
            for file_path in list(df_nfe_cur["File_Data"])
        ]
        # obter mes
        df_nfe_cur["Month"] = [str(dt)[:7] for dt in df_nfe_cur["Date"].values]
        df_nfe_cur = df_nfe_cur.sort_values(by="Date")


        # 2) Make copies to each folder and overwrite current ones
        lst_aux = ["xml", "pdf"]
        for i in range(len(df_nfe_cur)):
            nm = df_nfe_cur["Name"].values[i]
            month = df_nfe_cur["Month"].values[i]
            # xml
            src_file = df_nfe_cur["File_Data"].values[i]
            dst_file = f"{series_folder}/{year}/{month}/{nm}.xml"
            shutil.copy(src_file, dst_file)
            # pdf
            src_file = df_nfe_cur["File_Data_PDF"].values[i]
            dst_file = f"{series_folder}/{year}/{month}/{nm}.pdf"
            shutil.copy(src_file, dst_file)

        # 3) prepare data
        df_o = df_nfe_cur[['Date', 'Month', 'ValorServico', 'PTributoSN',
                           'ServicoID', 'Name', 'Prestador', 'Tomador', 'Projeto']].copy()

        df_o.rename(columns={
            "Date": "Data",
            "Month": "Mes",
            "ValorServico": "Valor",
            "PTributoSN": "Aliquota SN",
            "ServicoID": "Servico ID",
            "Name": "Codigo NFSe",
        }, inplace=True)

        df_o.to_csv(f"{series_folder}/{year}/receitas-declaradas_{year}.csv", sep=";", index=False)

        return df_o

def merge_files(file_prefix, output_name, series_folder, year):
    lst_pdfs = glob.glob(f"{series_folder}/{year}/*/{file_prefix}*.pdf")
    output_pdf = FileSys.merge_pdfs(
        lst_pdfs=lst_pdfs,
        dst_dir=f"{series_folder}/{year}",
        output_filename=f"{output_name}_{year}"
    )
    return output_pdf

def print_head():
    print("\n")
    print("*" * 50)


def main(src_root):
    # series
    series_folder = f"{src_root}/babylon/slu/series"

    # THE year
    year = str(last_month_year())

    print_head()
    print("\nWIN BABYLON 2")
    print(f"\nAssessed year: {year}")

    print_head()
    print("\nRun daily balance...")
    df = run_daily_balance(
        series_folder=series_folder,
        year=year
    )
    print("Daily Cashflow:\n")
    print(df)

    print_head()
    print("\nRun monthly balance...")
    dfo, dfm = run_monthly_balance(df, ano_especifico=year, series_folder=series_folder,
        year=year)
    print("Monthly Cashflow:\n")
    print(dfo)
    print("\nSummary:\n")
    print(dfm)

    print_head()
    print("\nRun declared revenue...")
    print("Declared revenue:")
    dfr = run_declared_revenue(
        src_root=src_root,
        series_folder=series_folder,
        year=year
    )
    print(dfr[["Data", "Valor", "Projeto", "Codigo NFSe"]].to_string())
    print("\nTotal declared revenue: R$ {}".format(dfr["Valor"].sum().round(2)))
    print("\n\n")

    print_head()
    print("\nArchive Babylon PDF files...")
    pdf1 = merge_files(
        file_prefix="NFSe_", output_name="receitas-declaradas",
        series_folder=series_folder,
        year=year
    )
    pdf2 = merge_files(file_prefix="dasn_", output_name="dasn",
    series_folder = series_folder,
    year = year
    )
    pdf3 = merge_files(file_prefix="darf_inss", output_name="darf_inss",
                       series_folder=series_folder,
                       year=year
                       )
    pdf4 = merge_files(file_prefix="extrato_cc", output_name="extrato_cc",
                       series_folder=series_folder,
                       year=year
                       )
    pdf5 = merge_files(file_prefix="recibo_prolabore", output_name="recibo_prolabore",
                       series_folder=series_folder,
                       year=year
                       )
    pdf6 = merge_files(file_prefix="cobranca_contabilidade", output_name="cobranca_contabilidade",
                       series_folder=series_folder,
                       year=year
                       )

    lst_pdfs = [pdf1, pdf2, pdf3, pdf4, pdf5, pdf6]
    # Remove None values using filter
    lst_pdfs_clean = list(filter(lambda x: x is not None, lst_pdfs))
    FileSys.merge_pdfs(
        lst_pdfs=lst_pdfs_clean,
        dst_dir=f"{series_folder}/{year}",
        output_filename=f"arquivo_contabil_{year}"
    )

    print("ok")
    print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process accounting documents")
    parser.add_argument("src_root", type=str, help="Source root folder")
    args = parser.parse_args()

    # Call the main function with parameters passed via command line
    main(args.src_root)