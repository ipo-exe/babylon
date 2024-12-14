from root import *
import pandas as pd
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

def plot_cashflow(df, folder, filename, dt_field="Date", val_field="Saldo", rev_field="Receitas", exp_field="Despesas", valac_field="Saldo Acumulado", ):
    plt.style.use("seaborn-v0_8")

    # Prepare data
    df = df[[dt_field, rev_field, exp_field, val_field]].copy()
    df[dt_field] = pd.to_datetime(df[dt_field])
    df[valac_field] = df[val_field].cumsum()
    df[rev_field + "_ac"] = df[rev_field].cumsum()
    df[exp_field + "_ac"] = df[exp_field].cumsum()

    # Set limits for plots based on value ranges
    vmaxx = max(abs(df[val_field].min()), df[val_field].max())
    vmaxx_ac = max(abs(df[exp_field + "_ac"].min()), df[rev_field + "_ac"].max())
    vmaxx_re = max(abs(df[exp_field].min()), df[rev_field].max())
    bar_width = pd.Timedelta(days=15)

    # Set up GridSpec
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], figure=fig,
                  wspace=0.4,
                  hspace=0.3,
                  left=0.15,
                  bottom=0.08,
                  top=0.95,
                  right=0.95
                  )

    # Subplot 1: Cumulative Balance Line
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df[dt_field], df[valac_field], marker='o', color='blue', linestyle='-', label='Saldo Acumulado')
    ax1.plot(df[dt_field], df[rev_field + "_ac"], marker='o', color='darkgreen', linestyle='-',
             label='Receitas Acumuladas')
    ax1.plot(df[dt_field], df[exp_field + "_ac"], marker='o', color='tab:red', linestyle='-',
             label='Despesas Acumuladas')
    ax1.set_ylabel('R$')
    ax1.set_title('Saldo Acumulado')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylim(-1.3 * vmaxx_ac, 1.3 * vmaxx_ac)
    ax1.set_xlim(df[dt_field].values[0] - (2*bar_width), df[dt_field].values[-1] + bar_width)
    ax1.legend()

    # Annotate only the last point for each line
    last_date = df[dt_field].iloc[-1]
    ax1.text(
        last_date, df[valac_field].iloc[-1] - (vmaxx_ac / 10),
        f"{round(df[valac_field].iloc[-1], 2)}", ha='center', va='top', fontsize=9, color='navy'
    )
    ax1.text(
        last_date, df[rev_field + "_ac"].iloc[-1] + (vmaxx_ac / 10),
        f"{round(df[rev_field + '_ac'].iloc[-1], 2)}", ha='center', va='bottom', fontsize=9, color='darkgreen'
    )
    ax1.text(
        last_date, df[exp_field + "_ac"].iloc[-1] - (vmaxx_ac / 10),
        f"{round(df[exp_field + '_ac'].iloc[-1], 2)}", ha='center', va='top', fontsize=9, color='tab:red'
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
    ax2.set_title('Saldo Mensal')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylim(-1.3 * vmaxx_re, 1.3 * vmaxx_re)
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

    # Subplot 3: Monthly Revenue and Expenses Bars
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(df[dt_field] - bar_width / 2, df[rev_field], width=bar_width, color='tab:green',)
    ax3.bar(df[dt_field] - bar_width / 2, df[exp_field], width=bar_width, color='tab:red',)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_ylabel('R$')
    ax3.set_title('Receitas e Despesas Mensais')
    ax3.set_ylim(-1.3 * vmaxx_re, 1.3 * vmaxx_re)
    ax3.set_xlim(df[dt_field].values[0] - (2*bar_width), df[dt_field].values[-1] + bar_width)
    for x, rev, exp in zip(df[dt_field], df[rev_field], df[exp_field]):
        ax3.text(x - bar_width / 2, rev + (vmaxx_re/20),
                 f"{'+ ' if rev >= 0 else '- '}{round(abs(rev), 2)}",
                 ha='center', va='bottom', fontsize=9, color='darkgreen')
        ax3.text(x - bar_width / 2, exp - (vmaxx_re/20),
                 f"{'+ ' if exp >= 0 else '- '}{round(abs(exp), 2)}",
                 ha='center', va='top', fontsize=9, color='darkred')

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

class NFe(DataSet):
    """
    Child class for handling NFSe XML data.
    """

    def __init__(self, name="NFSeDataSet", alias="NFeDS"):
        """
        Initialize the NFe object.
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
            'cnpj': tomador.find('.//default:CNPJ', ns).text,
            'nome': tomador.find('.//default:xNome', ns).text,
            'endereco': {
                'logradouro': tomador.find('.//default:end/default:xLgr', ns).text,
                'numero': tomador.find('.//default:end/default:nro', ns).text,
                'complemento': tomador.find('.//default:end/default:xCpl', ns).text if tomador.find('.//default:end/default:xCpl', ns) is not None else None,
                'bairro': tomador.find('.//default:end/default:xBairro', ns).text,
                'cidade': tomador.find('.//default:end/default:endNac/default:cMun', ns).text,
                'cep': tomador.find('.//default:end/default:endNac/default:CEP', ns).text
            }
        }

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
        self.taker = self.data[self.taker_field]["cnpj"] + " -- " + self.data[self.taker_field]["nome"]
        self.service_value = self.data[self.service_value_field]
        self.service_value_trib = nfse_data['servico']["p_tributo_SN"]
        self.service_id = self.data["servico"]["codigo_servico"]


class NFeColl(Collection):
    def __init__(self, base_object=NFe, name="MyNFeCollection", alias="NFeCol0"):
        """Initialize the ``NFeColl`` object.

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
        """Load NFe files from a folder

        :param folder: path to folder
        :type folder: str
        :return: None
        :rtype: None
        """
        from glob import glob
        lst_files = glob("{}/*.xml".format(folder))
        self.load_files(lst_files=lst_files)


    def load_files(self, lst_files):
        """Load NFe files from a list of files

        :param lst_files: list of paths to files
        :type lst_files: list
        :return: None
        :rtype: None
        """
        for f in lst_files:
            nfe_id = 'NFe_' + os.path.basename(f).split(".")[0]
            nfe = NFe(name=nfe_id, alias=nfe_id)
            nfe.load_data(file_data=f)
            self.append(new_object=nfe)






if __name__ == "__main__":
    f =r"C:\Users\Ipo\My Drive\athens\babylon\slu\series\caixa-mensal_cc-rf.csv"
    df = pd.read_csv(f, sep=";", parse_dates=["Data"])
    print(df.head())

    plot_cashflow(df=df, folder=r"C:\Users\Ipo\My Drive\athens\babylon\slu\series", filename="caixa-mensal_cc-rf", dt_field="Data")



