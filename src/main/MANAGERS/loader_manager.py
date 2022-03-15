from main.LOADERS.module_loader import ModuleLoader
from main.LOADERS.publication_loader import PublicationLoader
from main.LOADERS.module_loader_ha import ModuleLoaderHA
from main.LOADERS.publication_loader_ha import PublicationLoaderHA

class LOADER_SECTION():

    def load_modules(self) -> None:
        """
            Load modules from SQL server and serialize.
        """
        ModuleLoader().load_pymongo_db()

    def load_publications(self) -> None:
        """
            Load publications from pymongo database and serialize.
        """
        PublicationLoader().load_pymongo_db()
        
    def load_modules_ha(self) -> None:
        """
            Load modules from SQL server and serialize.
        """
        ModuleLoaderHA().load_pymongo_db()

    def load_publications_ha(self) -> None:
        """
            Load publications from pymongo database and serialize.
        """
        PublicationLoaderHA().load_pymongo_db()
