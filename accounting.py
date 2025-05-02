from root import *
import pandas as pd
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Portaria Interministerial MPS/MF nº 6
TABELA_INSS_2025 = [
    (1518.00, 0.075),
    (2793.88, 0.09),
    (4190.83, 0.12),
    (8157.41, 0.14)
]

# MEDIDA PROVISÓRIA Nº 1.294, DE 11 DE ABRIL DE 2025
TABELA_IRRF_2025 = [
    (2428.81, 2826.65, 0.075, 182.16),
    (2826.66, 3751.05, 0.15, 394.16),
    (3751.06, 4664.68, 0.225, 675.49),
    (4664.68, float('inf'), 0.275, 908.73)
]

def calcular_irrf(salario_bruto, tabela, deducao=0.0):
    """
    Calcula o IRRF (Imposto de Renda Retido na Fonte) com base nas faixas de tributação.

    Parâmetros:
    - salario_bruto (float): salário bruto mensal.
    - deducao (float): total de deduções mensais (ex: INSS, dependentes). Default é 0.0.

    Retorna:
    - imposto (float): valor do IRRF a ser pago no mês.

    """

    # Base de cálculo do IR
    base = salario_bruto - deducao

    # Se estiver abaixo da menor faixa, não há imposto
    if base <= 2428.80:
        return 0.0

    # Identifica a faixa aplicável
    for faixa in tabela:
        if faixa[0] <= base <= faixa[1]:
            aliquota = faixa[2]
            deduzir = faixa[3]
            imposto = base * aliquota - deduzir
            return round(imposto, 2)

    # Segurança — não deveria chegar aqui
    return 0.0

def calcular_inss(salario, tabela, empregado=True, f_autonomo=0.11):
    """
    Calcula o desconto de INSS de forma progressiva conforme as faixas e alíquotas definidas.

    Parâmetros:
    - salario (float): valor bruto do salário ou rendimento mensal.
    - tabela (list of tuples): lista de faixas progressivas no formato (limite_superior, alíquota),
      ordenadas de forma crescente.
    - empregado (bool): se True, aplica o cálculo progressivo por faixas (regime de contribuinte empregado);
      se False, aplica a alíquota fixa definida para autônomos.
    - f_autonomo (float): alíquota única aplicada a autônomos (padrão: 11%). O valor máximo de salário
      considerado é o teto da última faixa da tabela.

    Retorna:
    - desconto (float): valor total a ser descontado a título de INSS.

    Exemplo:
    >>> faixas_2024 = [
    ...     (1518.00, 0.075),      # até R$ 1.518,00 → 7,5%
    ...     (2793.88, 0.09),       # de R$ 1.518,01 até R$ 2.793,88 → 9%
    ...     (4190.83, 0.12),       # de R$ 2.793,89 até R$ 4.190,83 → 12%
    ...     (8157.41, 0.14)        # de R$ 4.190,84 até R$ 8.157,41 → 14%
    ... ]
    >>> calcular_inss(3000.00, faixas_2024)
    285.12
    >>> calcular_inss(9000.00, faixas_2024, empregado=False)
    897.32
    """
    # Valor acumulado do desconto progressivo
    if empregado:
        desconto = 0.0
        limite_inferior = 0.0  # Limite inferior da faixa (inicialmente zero)
        for limite_superior, aliquota in tabela:
            # Se o salário ultrapassa o limite superior da faixa atual
            if salario > limite_superior:
                base = limite_superior - limite_inferior
            else:
                base = max(0, salario - limite_inferior)
            desconto += base * aliquota
            if salario <= limite_superior:
                break
            limite_inferior = limite_superior

    else:
        salario_inss = salario
        if salario >= tabela[-1][0]:
            salario_inss = tabela[-1][0]
        desconto = salario_inss * f_autonomo

    return round(desconto, 2)


def plot_yearly_cashflow(df, folder, filename):
    plt.style.use("seaborn-v0_8")

    # Definição dos campos
    fields = {
        "val": "Saldo",
        "valac": "Saldo Acum",
        "rev": "Receitas",
        "revac": "Receitas Acum",
        "exp": "Desembolsos",
        "expac": "Desembolsos Acum",
        "dt": "Data"
    }

    # Prepara a coluna de datas
    df[fields["dt"]] = pd.to_datetime(df[fields["dt"]])

    # Definindo margens e limites para os gráficos
    v_margin = 1.5
    bar_width = pd.Timedelta(days=15)

    # Calcula os valores máximos para os gráficos
    vmaxx_ac = max(abs(df[fields["expac"]].min()), df[fields["revac"]].max())
    vmaxx_re = max(abs(df[fields["exp"]].min()), df[fields["rev"]].max())

    # Configuração do GridSpec para os subgráficos
    fig = plt.figure(figsize=(9, 12))
    gs = GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1], figure=fig,
                  wspace=0.4, hspace=0.5, left=0.15, bottom=0.08, top=0.95, right=0.95)

    # Função para adicionar rótulos aos gráficos
    def add_xticks(ax, dates):
        ax.set_xticks(dates)
        ax.set_xticklabels(dates.dt.strftime('%b'))

    # Função para adicionar uma linha de base no gráfico
    def add_baseline(ax, y_limit):
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylim(-v_margin * y_limit, v_margin * y_limit)

    # Função para anotar os pontos finais de cada linha
    def annotate_last_point(ax, x, y, text, color):
        ax.text(x, y, f"{text:.2f}", ha='center', va='bottom', fontsize=9, color=color)

    # Subplot 1: Cumulative Balance Line
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df[fields["dt"]], df[fields["valac"]], marker='o', color='blue', label='Saldo')
    ax1.plot(df[fields["dt"]], df[fields["revac"]], marker='o', color='darkgreen', label='Receitas')
    ax1.plot(df[fields["dt"]], df[fields["expac"]], marker='o', color='tab:red', label='Despesas')
    ax1.set_ylabel('R$')
    ax1.set_title('Saldo Acumulado')
    add_baseline(ax1, vmaxx_ac)
    add_xticks(ax1, df[fields["dt"]])

    # Anotar os pontos finais
    last_date = df[fields["dt"]].iloc[-1]
    annotate_last_point(ax1, last_date, df[fields["valac"]].iloc[-1] - (vmaxx_ac / 10), df[fields["valac"]].iloc[-1], 'navy')
    annotate_last_point(ax1, last_date, df[fields["revac"]].iloc[-1] + (vmaxx_ac / 10), df[fields["revac"]].iloc[-1], 'darkgreen')
    annotate_last_point(ax1, last_date, df[fields["expac"]].iloc[-1] - (vmaxx_ac / 10), df[fields["expac"]].iloc[-1], 'tab:red')

    # Subplot 2: Monthly Cash Flow Bars
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(df[fields["dt"]] - bar_width / 2, df[fields["val"]], width=bar_width,
                   color=df[fields["val"]].apply(lambda x: 'darkgreen' if x >= 0 else 'tab:red'))
    ax2.set_ylabel('R$')
    ax2.set_title('Saldo')
    add_baseline(ax2, vmaxx_re)
    add_xticks(ax2, df[fields["dt"]])

    # Anotar as barras
    for bar, cash_flow in zip(bars, df[fields["val"]]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (vmaxx_re / 20) if cash_flow >= 0 else bar.get_height() - (vmaxx_re / 20),
                 f"{'+ ' if cash_flow >= 0 else '- '}{abs(cash_flow):.2f}",
                 ha='center', va='bottom' if cash_flow >= 0 else 'top',
                 color='darkgreen' if cash_flow >= 0 else 'darkred', fontsize=9)

    # Linha da média
    v_mean = df[fields["val"]].mean()
    mean_color = 'green' if v_mean >= 0 else 'red'
    ax2.hlines(y=v_mean, xmin=df[fields["dt"]].values[0], xmax=df[fields["dt"]].values[-1], color=mean_color,
               linestyles="--", label=f"Média: {v_mean:.2f}", zorder=0)
    plt.legend()

    # Subplot 3: Monthly Revenue and Expenses Bars
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(df[fields["dt"]] - bar_width / 2, df[fields["rev"]], width=bar_width, color='tab:green')
    ax3.bar(df[fields["dt"]] - bar_width / 2, df[fields["exp"]], width=bar_width, color='tab:red')
    ax3.set_ylabel('R$')
    ax3.set_title('Receitas e Desembolsos')
    add_baseline(ax3, vmaxx_re)
    add_xticks(ax3, df[fields["dt"]])

    # Anotar as barras de receitas e despesas
    for x, rev, exp in zip(df[fields["dt"]], df[fields["rev"]], df[fields["exp"]]):
        ax3.text(x - bar_width / 2, rev + (vmaxx_re / 20), f"{'+ ' if rev >= 0 else '- '}{abs(rev):.2f}",
                 ha='center', va='bottom', fontsize=9, color='darkgreen')
        ax3.text(x - bar_width / 2, exp - (vmaxx_re / 20), f"{'+ ' if exp >= 0 else '- '}{abs(exp):.2f}",
                 ha='center', va='top', fontsize=9, color='darkred')
    '''
    # Linha da média para receitas
    rev_mean = df[fields["rev"]].mean()
    rev_mean_color = 'darkgreen' if rev_mean >= 0 else 'darkred'
    ax3.hlines(y=rev_mean, xmin=df[fields["dt"]].values[0], xmax=df[fields["dt"]].values[-1], color=rev_mean_color,
               linestyles="--", label=f"Média: {rev_mean:.2f}", zorder=0)

    # Linha da média para despesas
    exp_mean = df[fields["exp"]].mean()
    exp_mean_color = 'darkgreen' if exp_mean >= 0 else 'darkred'
    ax3.hlines(y=exp_mean, xmin=df[fields["dt"]].values[0], xmax=df[fields["dt"]].values[-1], color=exp_mean_color,
               linestyles="--", label=f"Média: {exp_mean:.2f}", zorder=0)

    plt.legend()
    '''

    # Subplot 4: Lucros
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.bar(df[fields["dt"]] - bar_width / 2, df["Lucros"], width=bar_width, color='magenta')
    ax4.set_ylabel('R$')
    ax4.set_title('Lucros')
    add_baseline(ax4, vmaxx_re)
    add_xticks(ax4, df[fields["dt"]])

    for x, rev in zip(df[fields["dt"]], df["Lucros"]):
        ax4.text(x - bar_width / 2, rev - (vmaxx_re / 3), f"{'+ ' if rev >= 0 else '- '}{abs(rev):.2f}",
                 ha='center', va='bottom', fontsize=9, color='purple')

    # Subplot 5: Prolabore
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.bar(df[fields["dt"]] - bar_width / 2, df["Prolabore"], width=bar_width, color='magenta')
    ax5.set_ylabel('R$')
    ax5.set_title('Prolabore')
    add_baseline(ax5, vmaxx_re)
    add_xticks(ax5, df[fields["dt"]])

    for x, rev in zip(df[fields["dt"]], df["Prolabore"]):
        ax5.text(x - bar_width / 2, rev - (vmaxx_re / 3), f"{'+ ' if rev >= 0 else '- '}{abs(rev):.2f}",
                 ha='center', va='bottom', fontsize=9, color='purple')

    # Salva o gráfico
    plt.savefig(f"{folder}/{filename}.jpg", dpi=400)

def _plot_yearly_cashflow(df, folder, filename):

    plt.style.use("seaborn-v0_8")

    val_field = "Saldo"
    valac_field = "Saldo Acum"
    rev_field = "Receitas"
    revac_field = "Receitas Acum"
    exp_field = "Desembolsos"
    expac_field = "Desembolsos Acum"
    dt_field = "Data"

    v_margin = 1.5

    df[dt_field] = pd.to_datetime(df[dt_field])


    # Set limits for plots based on value ranges
    vmaxx = max(abs(df[val_field].min()), df[val_field].max())
    vmaxx_ac = max(abs(df[expac_field].min()), df[revac_field].max())
    vmaxx_re = max(abs(df[exp_field].min()), df[rev_field].max())
    bar_width = pd.Timedelta(days=15)

    # Set up GridSpec
    fig = plt.figure(figsize=(9, 12))
    gs = GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1], figure=fig,
                  wspace=0.4,
                  hspace=0.5,
                  left=0.15,
                  bottom=0.08,
                  top=0.95,
                  right=0.95
                  )

    # Subplot 1: Cumulative Balance Line
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df[dt_field], df[valac_field], marker='o', color='blue', linestyle='-', label='Saldo')
    ax1.plot(df[dt_field], df[revac_field], marker='o', color='darkgreen', linestyle='-',
             label='Receitas')
    ax1.plot(df[dt_field], df[expac_field], marker='o', color='tab:red', linestyle='-',
             label='Desembolsos')
    ax1.set_ylabel('R$')
    ax1.set_title('Saldo Acumulado')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylim(-v_margin * vmaxx_ac, v_margin * vmaxx_ac)
    ax1.set_xlim(df[dt_field].values[0] - (2*bar_width), df[dt_field].values[-1] + bar_width)
    # Substituindo os rótulos do eixo X com as siglas dos meses
    plt.xticks(df[dt_field], df[dt_field].dt.strftime('%b'))
    ax1.legend()

    # Annotate only the last point for each line
    last_date = df[dt_field].iloc[-1]
    ax1.text(
        last_date, df[valac_field].iloc[-1] - (vmaxx_ac / 10),
        f"{round(df[valac_field].iloc[-1], 2)}", ha='center', va='top', fontsize=9, color='navy'
    )
    ax1.text(
        last_date, df[revac_field].iloc[-1] + (vmaxx_ac / 10),
        f"{round(df[revac_field].iloc[-1], 2)}", ha='center', va='bottom', fontsize=9, color='darkgreen'
    )
    ax1.text(
        last_date, df[expac_field].iloc[-1] - (vmaxx_ac / 10),
        f"{round(df[revac_field].iloc[-1], 2)}", ha='center', va='top', fontsize=9, color='tab:red'
    )

    # Subplot 2: Monthly Cash Flow Bars
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(
        df[dt_field] - bar_width / 2,
        df[val_field],
        width=bar_width,
        color=df[val_field].apply(lambda x: 'darkgreen' if x >= 0 else 'tab:red')
    )
    ax2.set_ylabel('R$')
    ax2.set_title('Saldo')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylim(-v_margin * vmaxx_re, v_margin * vmaxx_re)
    ax2.set_xlim(df[dt_field].values[0] - (2*bar_width), df[dt_field].values[-1] + bar_width)
    for bar, cash_flow in zip(bars, df[val_field]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (vmaxx_re / 20) if cash_flow >= 0 else bar.get_height() - (vmaxx_re / 20),
            f"{'+ ' if cash_flow >= 0 else '- '}{round(abs(cash_flow), 2)}",
            ha='center',
            va='bottom' if cash_flow >= 0 else 'top',
            color='darkgreen' if cash_flow >= 0 else 'darkred',
            fontsize=9
    )
    v_mean = df[val_field].mean()
    if v_mean >=0:
        c_mean = 'green'
    else:
        c_mean = 'red'
    ax2.hlines(
        y=v_mean,
        xmin=df[dt_field].values[0],
        xmax=df[dt_field].values[-1],
        color=c_mean,
        zorder=0,
        linestyles="--",
        label="Média: {}".format(round(v_mean, 2))
    )
    plt.legend()
    # Substituindo os rótulos do eixo X com as siglas dos meses
    plt.xticks(df[dt_field], df[dt_field].dt.strftime('%b'))

    # Subplot 3: Monthly Revenue and Expenses Bars
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(df[dt_field] - bar_width / 2, df[rev_field], width=bar_width, color='tab:green',)
    ax3.bar(df[dt_field] - bar_width / 2, df[exp_field], width=bar_width, color='tab:red',)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_ylabel('R$')
    ax3.set_title('Receitas e Desembolsos')
    ax3.set_ylim(-v_margin * vmaxx_re, v_margin * vmaxx_re)
    ax3.set_xlim(df[dt_field].values[0] - (2*bar_width), df[dt_field].values[-1] + bar_width)
    for x, rev, exp in zip(df[dt_field], df[rev_field], df[exp_field]):
        ax3.text(x - bar_width / 2, rev + (vmaxx_re/20),
                 f"{'+ ' if rev >= 0 else '- '}{round(abs(rev), 2)}",
                 ha='center', va='bottom', fontsize=9, color='darkgreen')
        ax3.text(x - bar_width / 2, exp - (vmaxx_re/20),
                 f"{'+ ' if exp >= 0 else '- '}{round(abs(exp), 2)}",
                 ha='center', va='top', fontsize=9, color='darkred')
    # Substituindo os rótulos do eixo X com as siglas dos meses
    plt.xticks(df[dt_field], df[dt_field].dt.strftime('%b'))

    # Subplot 4: Lucros
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.bar(df[dt_field] - bar_width / 2, df["Lucros"], width=bar_width, color='magenta', )
    ax4.axhline(0, color='black', linewidth=1)
    ax4.set_ylabel('R$')
    ax4.set_title('Lucros')
    ax4.set_ylim(-v_margin * vmaxx_re, v_margin * vmaxx_re)
    ax4.set_xlim(df[dt_field].values[0] - (2 * bar_width), df[dt_field].values[-1] + bar_width)
    for x, rev in zip(df[dt_field], df["Lucros"]):
        ax4.text(x - bar_width / 2, rev - (vmaxx_re/3),
                 f"{'+ ' if rev >= 0 else '- '}{round(abs(rev), 2)}",
                 ha='center', va='bottom', fontsize=9, color='purple')
    # Substituindo os rótulos do eixo X com as siglas dos meses
    plt.xticks(df[dt_field], df[dt_field].dt.strftime('%b'))

    # Subplot 4: Lucros
    ax4 = fig.add_subplot(gs[4, 0])
    ax4.bar(df[dt_field] - bar_width / 2, df["Prolabore"], width=bar_width, color='magenta', )
    ax4.axhline(0, color='black', linewidth=1)
    ax4.set_ylabel('R$')
    ax4.set_title('Prolabore')
    ax4.set_ylim(-v_margin * vmaxx_re, v_margin * vmaxx_re)
    ax4.set_xlim(df[dt_field].values[0] - (2 * bar_width), df[dt_field].values[-1] + bar_width)
    for x, rev in zip(df[dt_field], df["Prolabore"]):
        ax4.text(x - bar_width / 2, rev - (vmaxx_re / 3),
                 f"{'+ ' if rev >= 0 else '- '}{round(abs(rev), 2)}",
                 ha='center', va='bottom', fontsize=9, color='purple')
    # Substituindo os rótulos do eixo X com as siglas dos meses
    plt.xticks(df[dt_field], df[dt_field].dt.strftime('%b'))


    # Shared X-axis formatting and layout adjustment
    #plt.xlabel('Mês')
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.show()
    plt.savefig(f"{folder}/{filename}.jpg", dpi=400)

class Budget(RecordTable):

    def __init__(self, name="MyBudget", alias="Bud"):
        super().__init__(name=name, alias=alias)

        # ------------- specifics attributes ------------- #
        self.total_revenue = None
        self.total_expenses = None
        self.total_net = None
        self.summary_ascend = False

    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # set temporary util fields
        self.sign_field = "Sign"
        self.value_signed = "Value_Signed"
        # ... continues in downstream objects ... #

    def _set_data_columns(self):
        """Set specifics data columns names.
        Base Dummy Method. Expected to be incremented in superior methods.

        """
        # Main data columns
        self.columns_data_main = [
            "Type",
            "Status",
            "Contract",
            "Name",
            "Value",
        ]
        # Extra data columns
        self.columns_data_extra = [
            # Status extra
            "Date_Due",
            "Date_Exe",

            # Name extra
            # tags
            "Tags",

            # Values extra
            # Payment details
            "Method",
            "Protocol",
        ]
        # File columns
        self.columns_data_files = [
            "File_Receipt", "File_Invoice", "File_NF",
        ]
        # concat all lists
        self.columns_data = self.columns_data_main + self.columns_data_extra + self.columns_data_files

        # variations
        self.columns_data_status = self.columns_data_main + [
            self.columns_data_extra[0],
            self.columns_data_extra[1],
        ]

        # ... continues in downstream objects ... #

    def _set_operator(self):
        """Private method for Budget operator

        :return: None
        :rtype: None
        """

        # ------------- define sub routines here ------------- #
        def func_file_status():
            return FileSys.check_file_status(files=self.data["File"].values)

        def func_update_status():
            # filter relevante data
            df = self.data[["Status", "Method", "Date_Due"]].copy()
            # Convert 'Date_Due' to datetime format
            df['Date_Due'] = pd.to_datetime(self.data['Date_Due'])
            # Get the current date
            current_dt = datetime.datetime.now()

            # Update 'Status' for records with 'Automatic' method and 'Expected' status based on the condition
            condition = (df['Method'] == 'Automatic') & (df['Status'] == 'Expected') & (df['Date_Due'] <= current_dt)
            df.loc[condition, 'Status'] = 'Executed'

            # return values
            return df["Status"].values

        # todo implement all operations
        # ---------------- the operator ---------------- #

        self.operator = {
            "Status": func_update_status,
        }

    def _get_total_expenses(self, filter=True):
        filtered_df = self._filter_prospected_cancelled() if filter else self.data
        _n = filtered_df[filtered_df["Type"] == "Expense"]["Value_Signed"].sum()
        return round(_n, 3)

    def _get_total_revenue(self, filter=True):
        filtered_df = self._filter_prospected_cancelled() if filter else self.data
        _n = filtered_df[filtered_df["Type"] == "Revenue"]["Value_Signed"].sum()
        return round(_n, 3)

    def _filter_prospected_cancelled(self):
        return self.data[(self.data['Status'] != 'Prospected') & (self.data['Status'] != 'Cancelled')]

    def update(self):
        super().update()
        if self.data is not None:
            self.total_revenue = self._get_total_revenue(filter=True)
            self.total_expenses = self._get_total_expenses(filter=True)
            self.total_net = self.total_revenue + self.total_expenses
            if self.total_net > 0:
                self.summary_ascend = False
            else:
                self.summary_ascend = True

        # ... continues in downstream objects ... #
        return None

    def set_data(self, input_df):
        """Set RecordTable data from incoming dataframe.
        Expected to be incremented downstream.

        :param input_df: incoming dataframe
        :type input_df: dataframe
        :return: None
        :rtype: None
        """
        super().set_data(input_df=input_df)
        # convert to numeric
        self.data["Value"] = pd.to_numeric(self.data["Value"])
        # compute temporary field

        # sign and value_signed
        self.data["Sign"] = self.data["Type"].apply(lambda x: 1 if x == "Revenue" else -1)
        self.data["Value_Signed"] = self.data["Sign"] * self.data["Value"]

    def get_summary_by_type(self):
        summary = pd.DataFrame({
            "Total_Expenses": [self.total_expenses],
            "Total_Revenue": [self.total_revenue],
            "Total_Net": [self.total_net]
        })
        summary = summary.apply(lambda x: x.sort_values(ascending=self.summary_ascend), axis=1)
        return summary

    def get_summary_by_status(self, filter=True):
        filtered_df = self._filter_prospected_cancelled() if filter else self.data
        return filtered_df.groupby("Status")["Value_Signed"].sum().sort_values(ascending=self.summary_ascend)

    def get_summary_by_contract(self, filter=True):
        filtered_df = self._filter_prospected_cancelled() if filter else self.data
        return filtered_df.groupby("Contract")["Value_Signed"].sum().sort_values(ascending=self.summary_ascend)

    def get_summary_by_tags(self, filter=True):
        filtered_df = self._filter_prospected_cancelled() if filter else self.data
        tags_summary = filtered_df.groupby("Tags")["Value_Signed"].sum().sort_values(ascending=self.summary_ascend)
        tags_summary = tags_summary.sort()
        separate_tags_summary = filtered_df["Tags"].str.split(expand=True).stack().value_counts()
        print(type(separate_tags_summary))
        return tags_summary, separate_tags_summary

    @staticmethod
    def parse_annual_budget(year, budget_df, freq_field="Freq"):
        start_date = "{}-01-01".format(year)
        end_date = "{}-01-01".format(int(year) + 1)

        annual_budget = pd.DataFrame()

        for _, row in budget_df.iterrows():
            # Generate date range based on frequency
            dates = pd.date_range(start=start_date, end=end_date, freq=row['Freq'])

            # Replicate the row for each date
            replicated_data = pd.DataFrame({col: [row[col]] * len(dates) for col in df.columns})
            replicated_data['Date_Due'] = dates

            # Append to the expanded budget
            annual_budget = pd.concat([annual_budget, replicated_data], ignore_index=True)

        return annual_budget

class NFSe(DataSet):
    """
    Child class for handling NFSe XML data.
    """

    def __init__(self, name="NFSeDataSet", alias="NFSe"):
        """
        Initialize the NFSe object.
        """
        super().__init__(name=name, alias=alias)

        self.date = None
        self.emitter = None
        self.taker = None
        self.service_value = None
        self.service_value_trib = None
        self.service_id = None
        self.project_alias = None

    def __str__(self):
        """
        Nicely formatted string representation of the NFSe data.
        """
        if self.data is None:
            return "No data loaded."

        # Format the main NFSe data
        nfse_info = (
            f"NFSe ID: {self.data.get('nfse_id', 'N/A')}\n"
            f"Local de Emissão: {self.data.get('local_emissao', 'N/A')}\n"
            f"Local de Prestação: {self.data.get('local_prestacao', 'N/A')}\n"
            f"Número da NFSe: {self.data.get('numero_nfse', 'N/A')}\n"
            f"Código de Local de Incidência: {self.data.get('codigo_local_incidencia', 'N/A')}\n"
            f"Descrição do Serviço: {self.data.get('descricao_servico', 'N/A')}\n"
            f"Valor Líquido: {self.data.get('valor_liquido', 'N/A')}\n"
            f"Data do Processo: {self.data.get('data_processo', 'N/A')}\n"
            f"Data Competência: {self.date}\n"
        )

        # Format the emitente (issuer) information
        emitente = self.data.get(self.emitter_field, {})
        emitente_info = (
            f"Prestador:\n"
            f"  CNPJ: {emitente.get('cnpj', 'N/A')}\n"
            f"  Nome: {emitente.get('nome', 'N/A')}\n"
            f"  Endereço:\n"
            f"    Logradouro: {emitente.get('endereco', {}).get('logradouro', 'N/A')}\n"
            f"    Número: {emitente.get('endereco', {}).get('numero', 'N/A')}\n"
            f"    Bairro: {emitente.get('endereco', {}).get('bairro', 'N/A')}\n"
            f"    Cidade: {emitente.get('endereco', {}).get('cidade', 'N/A')}\n"
            f"    UF: {emitente.get('endereco', {}).get('uf', 'N/A')}\n"
            f"    CEP: {emitente.get('endereco', {}).get('cep', 'N/A')}\n"
            f"  Telefone: {emitente.get('telefone', 'N/A')}\n"
            f"  Email: {emitente.get('email', 'N/A')}\n"
        )

        # Format the tomador (receiver) information
        tomador = self.data.get(self.taker_field, {})
        tomador_info = (
            f"Tomador:\n"
            f"  CNPJ: {tomador.get('cnpj', 'N/A')}\n"
            f"  Nome: {tomador.get('nome', 'N/A')}\n"
            f"  Endereço:\n"
            f"    Logradouro: {tomador.get('endereco', {}).get('logradouro', 'N/A')}\n"
            f"    Número: {tomador.get('endereco', {}).get('numero', 'N/A')}\n"
            f"    Complemento: {tomador.get('endereco', {}).get('complemento', 'N/A')}\n"
            f"    Bairro: {tomador.get('endereco', {}).get('bairro', 'N/A')}\n"
            f"    Cidade: {tomador.get('endereco', {}).get('cidade', 'N/A')}\n"
            f"    CEP: {tomador.get('endereco', {}).get('cep', 'N/A')}\n"
        )

        # Format the service information
        servico = self.data.get('servico', {})
        servico_info = (
            f"Serviço:\n"
            f"  Código do Serviço: {servico.get('codigo_servico', 'N/A')}\n"
            f"  Descrição: {servico.get('descricao_servico', 'N/A')}\n"
            f"  Valor do Serviço: {servico.get('valor_servico', 'N/A')}\n"
        )

        # Combine all sections into one string
        return f"{nfse_info}\n{emitente_info}\n{tomador_info}\n{servico_info}"

    def _set_fields(self):
        """Set fields names.
        Expected to increment superior methods.

        """
        # ------------ call super ----------- #
        super()._set_fields()
        # Attribute fields
        self.date_field = "Date"
        self.emitter_field = "Prestador"
        self.taker_field = "Tomador"
        self.service_value_field = "ValorServico"
        self.service_value_trib_field = "PTributoSN"
        self.service_id_field = "ServicoID"
        self.project_alias_field = "Projeto"


        # ... continues in downstream objects ... #

    def get_metadata(self):
        """Get a dictionary with object metadata.
        Expected to increment superior methods.

        .. note::

            Metadata does **not** necessarily inclue all object attributes.

        :return: dictionary with all metadata
        :rtype: dict
        """
        # ------------ call super ----------- #
        dict_meta = super().get_metadata()

        # customize local metadata:
        dict_meta_local = {
            self.date_field: self.date,
            self.emitter_field: self.emitter,
            self.taker_field: self.taker,
            self.service_value_field: self.service_value,
            self.service_value_trib_field: self.service_value_trib,
            self.service_id_field: self.service_id,
            self.project_alias_field: self.project_alias,
        }

        # update
        dict_meta.update(dict_meta_local)
        return dict_meta

    def load_data(self, file_data):
        """Load and parse XML data from the provided file.

        :param file_data: file path to the NFSe XML data.
        :type file_data: str
        :return: None
        """
        # Ensure the file path is absolute
        file_data = os.path.abspath(file_data)
        #print(file_data)
        tree = ET.parse(file_data)
        root = tree.getroot()

        # Namespaces used in the XML
        ns = {
            'default': 'http://www.sped.fazenda.gov.br/nfse',
            'ds': 'http://www.w3.org/2000/09/xmldsig#'
        }

        # Dictionary to hold extracted XML data
        nfse_data = {}

        # Extract main NFSe data
        nfse_data['nfse_id'] = root.find('.//default:infNFSe', ns).attrib.get('Id')
        nfse_data['local_emissao'] = root.find('.//default:xLocEmi', ns).text
        nfse_data['local_prestacao'] = root.find('.//default:xLocPrestacao', ns).text
        nfse_data['numero_nfse'] = root.find('.//default:nNFSe', ns).text
        nfse_data['codigo_local_incidencia'] = root.find('.//default:cLocIncid', ns).text
        nfse_data['descricao_servico'] = root.find('.//default:xTribNac', ns).text
        nfse_data['valor_liquido'] = float(root.find('.//default:vLiq', ns).text)
        nfse_data['data_processo'] = root.find('.//default:dhProc', ns).text
        nfse_data[self.date_field] = root.find('.//default:dCompet', ns).text

        # Extract emitente (issuer) data
        emitente = root.find('.//default:emit', ns)
        nfse_data[self.emitter_field] = {
            'cnpj': emitente.find('.//default:CNPJ', ns).text,
            'nome': emitente.find('.//default:xNome', ns).text,
            'endereco': {
                'logradouro': emitente.find('.//default:enderNac/default:xLgr', ns).text,
                'numero': emitente.find('.//default:enderNac/default:nro', ns).text,
                'bairro': emitente.find('.//default:enderNac/default:xBairro', ns).text,
                'cidade': emitente.find('.//default:enderNac/default:cMun', ns).text,
                'uf': emitente.find('.//default:enderNac/default:UF', ns).text,
                'cep': emitente.find('.//default:enderNac/default:CEP', ns).text
            },
            'telefone': emitente.find('.//default:fone', ns).text,
            'email': emitente.find('.//default:email', ns).text
        }

        # Extract tomador (receiver) data
        tomador = root.find('.//default:toma', ns)

        nfse_data[self.taker_field] = {
            'cnpj': tomador.find('.//default:CNPJ', ns).text if tomador.find('.//default:CNPJ', ns) is not None else None,
            'nif': tomador.find('.//default:NIF', ns).text if tomador.find('.//default:NIF',
                                                                           ns) is not None else None,
            'nome': tomador.find('.//default:xNome', ns).text,
        }
        #print()
        #print(nfse_data[self.taker_field]["nome"])
        _address =    {
                'logradouro': tomador.find('.//default:end/default:xLgr', ns).text,
                'numero': tomador.find('.//default:end/default:nro', ns).text,
                'complemento': tomador.find('.//default:end/default:xCpl', ns).text if tomador.find('.//default:end/default:xCpl', ns) is not None else None,
                'bairro': tomador.find('.//default:end/default:xBairro', ns).text if tomador.find('.//default:end/default:xBairro', ns) is not None else None,
                'cidade': tomador.find('.//default:end/default:endNac/default:cMun', ns).text if tomador.find('.//default:end/default:endNac/default:cMun', ns) is not None else None,
                'cep': tomador.find('.//default:end/default:endNac/default:CEP', ns).text if tomador.find('.//default:end/default:endNac/default:CEP', ns) is not None else None,
        }

        nfse_data[self.taker_field]['endereco'] = _address.copy()

        # Extract service data
        servico = root.find('.//default:serv', ns)
        nfse_data['servico'] = {
            'codigo_servico': servico.find('.//default:cServ/default:cTribNac', ns).text,
            'descricao_servico': servico.find('.//default:cServ/default:xDescServ', ns).text,
        }
        valor_servico_element = root.find('.//default:valores/default:vServPrest/default:vServ', ns)
        nfse_data[self.service_value_field] = float(valor_servico_element.text)
        nfse_data['servico']["valor_servico"] = nfse_data[self.service_value_field]
        tribut_element = root.find('.//default:valores/default:trib/default:totTrib/default:pTotTribSN', ns)
        if tribut_element is None:
            # Handle
            v_tb = 6.0
        else:
            v_tb = float(str(tribut_element.text))
        nfse_data['servico']["p_tributo_SN"] = v_tb

        # Set parsed data to the class attribute
        self.data = nfse_data
        self.date = nfse_data[self.date_field]
        self.file_data = file_data
        self.emitter = self.data[self.emitter_field]["cnpj"] + " -- " + self.data[self.emitter_field]["nome"]
        # hande NIF or CNPJ
        if self.data[self.taker_field]["cnpj"] is not None:
            self.taker = self.data[self.taker_field]["cnpj"] + " (CNPJ) -- " + self.data[self.taker_field]["nome"]
        else:
            self.taker = self.data[self.taker_field]["nif"] + " (NIF) -- " + self.data[self.taker_field]["nome"]
        self.service_value = self.data[self.service_value_field]
        self.service_value_trib = nfse_data['servico']["p_tributo_SN"]
        self.service_id = self.data["servico"]["codigo_servico"]


class NFSeColl(Collection):
    def __init__(self, base_object=NFSe, name="MyNFeCollection", alias="NFeCol0"):
        """Initialize the ``NFSeColl`` object.

        :param base_object: ``MbaE``-based object for collection
        :type base_object: :class:`MbaE`

        :param name: unique object name
        :type name: str

        :param alias: unique object alias.
            If None, it takes the first and last characters from name
        :type alias: str

        """
        # ------------ set pseudo-static ----------- #
        self.object_alias = "NFE_COL"
        # Set the name and baseobject attributes
        self.baseobject = base_object
        self.baseobject_name = base_object.__name__

        # Initialize the catalog with an empty DataFrame
        dict_metadata = self.baseobject().get_metadata()

        self.catalog = pd.DataFrame(columns=dict_metadata.keys())

        # Initialize the ``Collection`` as an empty dictionary
        self.collection = dict()

        # ------------ set mutables ----------- #
        self.size = 0

        self._set_fields()
        # ... continues in downstream objects ... #

    def load_folder(self, folder):
        """Load NFSe files from a folder

        :param folder: path to folder
        :type folder: str
        :return: None
        :rtype: None
        """
        from glob import glob
        lst_files = glob("{}/*.xml".format(folder))
        self.load_files(lst_files=lst_files)


    def load_files(self, lst_files):
        """Load NFSe files from a list of files

        :param lst_files: list of paths to files
        :type lst_files: list
        :return: None
        :rtype: None
        """
        for f in lst_files:
            nfe_id = 'NFSe_' + os.path.basename(f).split(".")[0]
            nfe = NFSe(name=nfe_id, alias=nfe_id)
            nfe.load_data(file_data=f)
            self.append(new_object=nfe)






if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    s = 5000
    d = calcular_inss(s, tabela=TABELA_INSS_2025, empregado=True)
    irrf = calcular_irrf(salario_bruto=s, tabela=TABELA_IRRF_2025, deducao=d)
    print(irrf)
    v = np.linspace(1500, 50000, num=200)
    #print(v)
    ls_ir = list()
    ls_ir_p = list()
    for i in range(len(v)):
        s = v[i]
        d = calcular_inss(s, tabela=TABELA_INSS_2025, empregado=True)
        irrf = calcular_irrf(salario_bruto=s, tabela=TABELA_IRRF_2025, deducao=d)
        #print(irrf)
        ls_ir.append(irrf)
        ls_ir_p.append(irrf / s)
    y = np.array(ls_ir)
    p = np.array(ls_ir_p) * 100
    plt.plot(v, p)
    plt.xlabel("R$ Renda Mensal")
    plt.ylabel("% de Imposto de Renda")
    plt.show()




