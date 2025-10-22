"""
La siguiente clase esta destinada a limpiar los 
datos de la base de datos covalto, en base a lo descubierto en el
en el notebook exploratorio.
"""


class CovaltoCsvDataCleane:
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from statsmodels.stats.stattools import medcouple

    """
    Clase encargada de limpiar los datos provenientes de un archivo csv de la 
    base de datos de covalto.

    Attributes
    ----------
    file_path: str or Path
        Ruta del archivo csv a limpiar
    """

    def __init__(self,file_path):

        """
        función que inicializa la clase con la ruta del archivo y verifica que
        el archivo exista.
        
        Attributes
        ----------
        file_path: Ruta del archivo csv a limpiar

        Raises
        ------
        FileNotFoundError
            Si el archivo específicado no existe

        """

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {self.file_path}")
        
        self.data = None                    
        

    
    def load_dataset(self):

        """
        Lee el archivo csv indicado en el self.file_path y lo carga como un
        dataframe de pandas.

        return
        ------
        pd.DataFrame
            DataFrame que contiene los datos leidos del archivo csv.

        Raises
        ------
        pandas.errors.EmptyDataError
            Si el archivo está vacío

        pandas.errors.ParserError
            Si el archivo CSV tiene errores de formato o no puede ser parseado correctamente.
        """

        self.data = pd.read_csv(self.file_path)
        return self.data
    
    
    def remove_outliers_medcouple(self, col_interes='ingresos_anuales_mxn',max_retries=3):

        """
        Elimina los valores atípicos de una columna numérica usando el estadístico MedCouple (MC).

        Este método ajusta los límites inferior y superior de detección de outliers 
        en función de la asimetría de la distribución, medida con el estadístico MedCouple.

        Parameters
        ----------
        covalto_dataset_raw: pd.DataFrame
            DataFrame que contiene los datos a procesar.

        col_interes: str 
            Nombre de la columna sobre la cual se eliminaran los valores atípicos.
            Por defecto es 'ingresos_anuales_mxn'.

        max_retries : int
            Número máximo de intentos para ingresar una columna válida.
            Por defecto es 3.
        
        Return
        ------
        pandas.DataFrame
            Copia del DataFrame original sin los valores atípicos en la columna indicada.

        Raises
        ------
        ValueError
            Si no se proporciona una columna válida después de varios intentos.
        
        Notes
        -----
            - Si el MedCouple (MC) es positivo, se ajusta más el límite inferior.
            - Si el MC es negativo, se ajusta más el límite superior.
            - Si el MC es 0, se aplica la regla estándar del rango intercuartílico (IQR).

        """

        if self.data is None:
            raise ValueError("El dataset no ha sido cargado. Ejecute load_dataset() primero.")
        
        attempt = 0
        while attempt<max_retries:
            try:
                # Intenta acceder a la columna
                data = self.data.copy()
                _=data[col_interes]

                # Calcular estadísticas
                resumen = data[col_interes].describe()
                Q1, Q2, Q3 = resumen.iloc[4], resumen.iloc[5], resumen.iloc[6]
                RI = Q3 - Q1
                MC = medcouple(data[col_interes].to_numpy())

                # Determinar límites
                if MC > 0:
                    lim_inf_MC = Q1 - 1.5 * np.exp(-3.5 * MC) * RI
                    lim_sup_MC = Q3 + 1.5 * np.exp(4 * MC) * RI
                elif MC < 0:
                    lim_inf_MC = Q1 - 1.5 * np.exp(-4 * MC) * RI
                    lim_sup_MC = Q3 + 1.5 * np.exp(3.5 * MC) * RI
                else:
                    lim_inf_MC = Q1 - 1.5 * RI
                    lim_sup_MC = Q3 + 1.5 * RI

                data = data[
                    (data[col_interes] >= lim_inf_MC) &
                    (data[col_interes] <= lim_sup_MC) |
                    (data[col_interes].isna())
                ]
                self.data = data.copy()
                return self.data
            
            except KeyError:
                print(f"La columna '{col_interes}' no existe en el DataFrame.")
                attempt += 1
                if attempt < max_retries:
                    col_interes = input("Por favor, ingrese el nombre correcto de la columna: ")
                else:
                    raise ValueError("No se proporcionó una columna válida después de varios intentos.")

    def standardize_sector_column(self):
        """
        Estandariza los nombres del sector industrial en el DataFrame.
        
        Reemplaza valores inconsistentes o en minúsculas por versiones normalizadas.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame que contiene la columna 'sector_industrial'.
        
        Returns
        -------
        pandas.DataFrame
            Copia del DataFrame con los nombres del sector estandarizados.
        """
        
        if 'sector_industrial' not in self.data.columns:
            raise ValueError("La columna 'sector_industrial' no existe en el dataset proporcionado.")
        
        self.data['sector_industrial'] = self.data['sector_industrial'].replace({
            'retail': 'Retail'})
        
        return self.data

