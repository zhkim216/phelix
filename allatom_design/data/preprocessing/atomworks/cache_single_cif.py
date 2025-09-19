from atomworks.io.parser import parse

    
def cache_single_cif(cif_path: str, cache_dir: str):
    
    _ = parse(
        # Caching arguments
        load_from_cache=False,
        save_to_cache=True,
        cache_dir=cache_dir,
        # Standard arguments
        filename=cif_path,
        build_assembly="all",
    )
    
    
if __name__ == "__main__":
    cache_single_cif(cif_path= "/home/possu/jinho/allatom-design/atomworks/tests/data/pdb/01/101m.cif.gz", cache_dir="/home/possu/jinho/allatom-design/atomworks_test/250917")