import glob
import os
import shutil
import argparse
from root import FileSys
from datetime import datetime

def add_prefix_to_filename(file_path, prefix):
    # Extract the base name and extension
    base_name = os.path.basename(file_path)
    # Create the new file name by adding the prefix before the file name
    new_file_path = os.path.join(os.path.dirname(file_path), prefix + base_name)
    # Rename the file
    os.rename(file_path, new_file_path)
    return new_file_path

def print_head():
    print("\n")
    print("*" * 50)

def main(src_root, print_bay, portfolio_bay):
    # Get the current date in YYYY-MM-DD format
    _now = datetime.now()
    today = str(_now.strftime("%Y-%m-%d-%H:%M:%S")).replace(":", "-")

    kinds = {
        "admin": [
            "proposta", "proposal", "contrato", "contract", "misc",
            "paperwork", "email", "reuniao", "meeting",
        ],
        "budget": ["misc", "recibo", "receipt"],
        "outputs": [
            "misc", "relatorio", "report", "palestra", "lecture", "art",
            "paper", "artigo", "conference-paper", "monograph", "blog-post", "certificado", "certificate",
        ]
    }

    c_kinds = {
        "admin": "pb",
        "budget": "pb",
        "outputs": "color"
    }

    portfolio_subs = {
        "palestra": "lectures", "lecture": "lectures", "art": "arts", "paper": "papers", "artigo": "papers",
        "conference-paper": "conference-papers", "monograph": "monographs", "blog-post": "blog-posts",
        "certificado": "certificates", "certificate": "certificates"
    }

    # pRINTER LOOP
    print_head()
    print("PRINTER")
    print(f"Timestamp: {today}")
    for k in kinds:
        dst_c = c_kinds[k]
        print_dst = f"{print_bay}/{dst_c}"
        for subkind in kinds[k]:
            lst_files = glob.glob(f"{src_root}/*/*/{k}/{subkind}_*.pdf")
            for f in lst_files:
                fnm = os.path.basename(f)
                print(f"PRINTER --- {k} -- {subkind} -- {fnm}")
                # copy to print bay
                shutil.copy(
                    src=f,
                    dst=f"{print_dst}/tmp/{fnm}"
                )
                # also copy to portfolio bay
                if k == "outputs" and subkind in list(portfolio_subs.keys()):
                    _d = portfolio_subs[subkind]
                    portd = f"{portfolio_bay}/{_d}"
                    shutil.copy(
                        src=f,
                        dst=f"{portd}/{fnm}"
                    )
                # rename file
                add_prefix_to_filename(
                    file_path=f,
                    prefix="__"
                )

    print("\n\nMerging PB files...")
    lst_files = glob.glob(f"{print_bay}/pb/tmp/*.pdf")
    FileSys.merge_pdfs(
        lst_pdfs=lst_files,
        dst_dir=f"{print_bay}/pb",
        output_filename=f"lote_pb_{today}"
    )
    print("Cleaning bay...")
    for f in lst_files:
        os.remove(f)
    print("\n\nMerging Color files...")
    lst_files = glob.glob(f"{print_bay}/color/tmp/*.pdf")
    FileSys.merge_pdfs(
        lst_pdfs=lst_files,
        dst_dir=f"{print_bay}/color",
        output_filename=f"lote_color_{today}"
    )
    print("Cleaning bay...")
    for f in lst_files:
        os.remove(f)
    print("ok\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folders for printing and portfolio management")
    parser.add_argument("src_root", type=str, help="Source root folder")
    parser.add_argument("print_bay", type=str, help="Print bay folder")
    parser.add_argument("portfolio_bay", type=str, help="Portfolio bay folder")

    args = parser.parse_args()

    # Call the main function with parameters passed via command line
    main(args.src_root, args.print_bay, args.portfolio_bay)
