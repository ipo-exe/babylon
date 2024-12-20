from root import FileSys
import os
import argparse
from datetime import datetime

def print_head():
    print("\n")
    print("*" * 50)

def main(backup_bay, vault_dir, data_dir):
    print_head()
    print("BACKUP")
    # Get the current date in YYYY-MM-DD format
    _now = datetime.now()
    today = str(_now.strftime("%Y-%m-%d-%H:%M:%S")).replace(":", "-")

    # vault backup
    vault_bay = f"{backup_bay}/vault"
    lst_dir = os.listdir(vault_bay)
    if len(lst_dir) == 0:
        print("vault backup --- run backup")
        FileSys.archive_folder(
            src_dir=vault_dir,
            dst_dir=vault_bay
        )

    else:
        print("vault backup --- SKIPPING: PENDING FILES IN BAY")


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup important folders")
    parser.add_argument("backup_bay", type=str, help="Backup bay folder")
    parser.add_argument("vault_dir", type=str, help="Source vault folder")
    parser.add_argument("data_dir", type=str, help="Source data folder")
    args = parser.parse_args()

    # Call the main function with parameters passed via command line
    main(args.backup_bay, args.vault_dir, args.data_dir)