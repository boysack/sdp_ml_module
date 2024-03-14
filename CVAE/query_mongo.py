from pymongo import MongoClient # only wokrs on python 3.9 and 3.10, NOT 3.11
from bson import ObjectId
from datetime import date
import time as pytime
import pandas as pd
from pprint import pprint
import json

# Connessione al client MongoDB (modifica l'URI se necessario)
client = MongoClient('mongodb://localhost:27017/')
db = client['arol']
db_eqtq = db['eqtq']
db_drive = db['drive']
db_plc = db['plc']

eqtq_sensors = ['AverageFriction', 'MaxLockPosition', 'MinLockPosition', 'LockDegree', 'AverageTorque', 'stsBadClosure', 'stsTotalCount', 'stsClosureOK', 'stsNoLoad']
plc_sensors = ['PowerCurrent', 'MainMotorCurrent', 'PowerVoltage', 'HeadMotorCurrent', 'OperationState', 'MainMotorSpeed', 'LubeLevel', 'AirConsumption', 'TotalProduct', 'ProdSpeed', 'OperationMode', 'HeadMotorSpeed', 'test2', 'Alarm', 'test1', 'test3']
drive_sensors = ['Tcpu', 'Tboard', 'Twindings', 'Tplate']
heads = ['Head_01', 'Head_02', 'Head_03', 'Head_04', 'Head_05', 'Head_06', 'Head_07', 'Head_08', 'Head_09', 'Head_10', 'Head_11', 'Head_12', 'Head_13', 'Head_14', 'Head_15', 'Head_16', 'Head_17', 'Head_18', 'Head_19', 'Head_20', 'Head_21', 'Head_22', 'Head_23', 'Head_24']

eqtq_data = []
plc_data = []
drive_data = []

def query_mongo(dati: list, frequency: int):
    day = date.today().strftime("%a %b %d %Y") # Ottieni la data odierna
    first_time = int(pytime.time()) * 1000 # Ottieni il timestamp attuale in millisecondi
    eqtq_query = []
    drive_query = []
    plc_query = []
    first_head = True

    for head in dati:
        # print(head)
        eqtq_samples = [] # Inizializza la lista dei campioni
        drive_samples = []
        plc_samples = []

        folder = heads[head[1] - 1] # 'Head_xx'
        nsamples = len(head[0]) # Calcola il numero di campioni
        nsamples_query = len(head[0]) # Calcola il numero di campioni
        for row in range(nsamples):
            if (row + 1) % frequency == 0:
                for col in range(len(head[0].columns)): # Per ogni colonna del dataframe 
                    if head[0].columns[col] in eqtq_sensors:
                        name = str(folder.replace('Head_', 'H') + '_' + head[0].columns[col])
                        value = float(head[0].iloc[row, col])
                        time = first_time + row * 1000
                        eqtq_samples.append(
                            {
                                "name": name,
                                "value": value,
                                "time": float(time)
                            }
                        )
                        
                    elif head[0].columns[col] in drive_sensors:
                        name = str(folder.replace('Head_', 'H') + '_' + head[0].columns[col])
                        value = float(head[0].iloc[row, col])
                        time = first_time + row * 1000
                        drive_samples.append(
                            {
                                "name": name,
                                "value": value,
                                "time": float(time)
                            }
                        )
        
        if len(eqtq_samples) > 0:
            eqtq_query.append(
            {
                "day": str(day),
                "folder": folder,
                "first_time": float(first_time),
                "last_time": eqtq_samples[-1]["time"],
                "nsamples": len(eqtq_samples),
                "samples": eqtq_samples
            }
            )

        if len(drive_samples) > 0:
            drive_query.append(
            {
                "day": str(day),
                "folder": folder,
                "first_time": float(first_time),
                "last_time": drive_samples[-1]["time"],
                "nsamples": len(drive_samples),
                "samples": drive_samples
            }
            )

        if first_head:  # Se è il primo head, crea i dati per plc
            first_head = False
            for col in range(len(head[0].columns)): # Per ogni colonna del dataframe
                variable = head[0].columns[col] # Ottieni il nome della colonna
                plc_samples = [] # Inizializza la lista dei campioni
                if variable in plc_sensors: # Se la variabile è un sensore PLC
                    for row in range(nsamples): # Per ogni riga della colonna interessata
                        if (row + 1) % frequency == 0:
                            value = float(head[0].iloc[row, col])
                            time = first_time + row * 1000
                            plc_samples.append(
                                {
                                    "value": value,
                                    "time": float(time),
                                }
                            )

                    if len(plc_samples) > 0:
                        plc_query.append(
                            {
                                "day": str(day),
                                "variable": variable,
                                "first_time": float(first_time),
                                "last_time": plc_samples[-1]["time"],
                                "nsamples": len(plc_samples),
                                "samples": plc_samples
                            }
                        )

    # Inserisci i dati nel database
    # se le query sono vuote, non fare nulla
    if eqtq_query:
        db_eqtq.insert_many(eqtq_query)
    if drive_query:
        db_drive.insert_many(drive_query)
    if plc_query:
        db_plc.insert_many(plc_query)
    # pprint(eqtq_query)
    # pprint(drive_query)
    # pprint(plc_query)
    