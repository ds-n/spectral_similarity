import numpy as np
import pandas as pd
from koinapy import Koina
import time


# TODO move this to make_predictions.intensity_predictions
def safe_obtain_predictions(
    peptides_batch, switched, max_retries=3, delay=1, model="Prosit_2020_intensity_HCD"
):
    """
    Attempts to obtain predictions for a batch of peptides, retrying in case of failure.

    Args:
        peptides_batch (list): List of peptide sequences.
        switched (bool): Parameter for prediction function.
        max_retries (int): Maximum number of retries before raising the exception.
        delay (int): Delay in seconds between retries.

    Returns:
        pd.DataFrame: DataFrame of predictions.
    """
    for attempt in range(max_retries):
        try:
            return obtain_predictions_pairs(
                peptides_batch,
                charges=np.array(len(peptides_batch) * [2]),
                switched=switched,
                model=model,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Prediction failed after {max_retries} attempts."
                ) from e


def safe_obtain_ccs_predictions(
    peptides_batch,
    switched,
    charges=None,
    max_retries=3,
    delay=1,
    model="AlphaPept_ccs_generic",
):
    """
    Attempts to obtain CCS/ion mobility predictions for a batch of peptides, retrying in case of failure.

    Args:
        peptides_batch (list or np.array): List of peptide sequences.
        switched (bool): Whether these are I/L-swapped sequences.
        charges (list or np.array, optional): Charge states. Defaults to 2 for all peptides.
        max_retries (int): Maximum number of retries before raising the exception.
        delay (int): Delay in seconds between retries.
        model (str): Koina model name for CCS prediction. Options include:
            - "AlphaPept_ccs_generic" (AlphaPeptDeep CCS model)

    Returns:
        pd.DataFrame: DataFrame of CCS predictions.
    """
    if charges is None:
        charges = np.array(len(peptides_batch) * [2])

    for attempt in range(max_retries):
        try:
            return obtain_ccs_predictions(
                peptides_batch,
                charges=charges,
                switched=switched,
                model=model,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"CCS prediction failed after {max_retries} attempts."
                ) from e


def safe_obtain_rt_predictions(
    peptides_batch,
    switched,
    max_retries=3,
    delay=1,
    model="Deeplc_hela_hf",
):
    """
    Attempts to obtain retention time predictions for a batch of peptides, retrying in case of failure.

    Args:
        peptides_batch (list or np.array): List of peptide sequences.
        switched (bool): Whether these are I/L-swapped sequences.
        max_retries (int): Maximum number of retries before raising the exception.
        delay (int): Delay in seconds between retries.
        model (str): Koina model name for RT prediction. Options include:
            - "Deeplc_hela_hf" (DeepLC trained on HeLa HF data)
            - "Prosit_2019_irt"
            - "AlphaPept_rt_generic"

    Returns:
        pd.DataFrame: DataFrame of retention time predictions.
    """
    for attempt in range(max_retries):
        try:
            return obtain_rt_predictions(
                peptides_batch,
                switched=switched,
                model=model,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"RT prediction failed after {max_retries} attempts."
                ) from e


def obtain_rt_predictions(
    peptides,
    switched=False,
    model="Deeplc_hela_hf",
):
    """
    Function to obtain retention time predictions for a set of peptides.

    Args:
        peptides (np.array): Array of peptide sequences.
        switched (bool): Whether these are I/L-swapped sequences.
        model (str): Koina model name. Options:
            - "Deeplc_hela_hf" - DeepLC trained on HeLa HF data
            - "Prosit_2019_irt" - Prosit iRT model
            - "AlphaPept_rt_generic" - AlphaPeptDeep RT model

    Returns:
        pd.DataFrame: DataFrame with peptide sequences and predicted retention times.
    """
    inputs = pd.DataFrame()
    inputs["peptide_sequences"] = np.array(peptides)
    inputs.drop_duplicates(inplace=True)
    print(f"RT prediction input shape: {inputs.shape}")

    koina_model = Koina(model, "koina.wilhelmlab.org:443")
    try:
        predictions = koina_model.predict(inputs, debug=True)
    except Exception as e:
        print(f"RT prediction error: {e}")
        raise

    predictions["non_switched"] = switched

    return predictions


def obtain_ccs_predictions(
    peptides,
    charges=None,
    switched=False,
    model="AlphaPept_ccs_generic",
):
    """
    Function to obtain CCS (collision cross section) / ion mobility predictions for a set of peptides.

    Args:
        peptides (np.array): Array of peptide sequences.
        charges (np.array, optional): Array of charge states. Defaults to 2 for all.
        switched (bool): Whether these are I/L-swapped sequences.
        model (str): Koina model name. Options:
            - "AlphaPept_ccs_generic" - AlphaPeptDeep CCS model

    Returns:
        pd.DataFrame: DataFrame with peptide sequences, charges, and predicted CCS values.
    """
    num_peptides = len(peptides)

    if charges is None:
        charges = np.array(num_peptides * [2])

    inputs = pd.DataFrame()
    inputs["peptide_sequences"] = np.array(peptides)
    inputs["precursor_charges"] = np.array(charges)
    inputs.drop_duplicates(inplace=True)
    print(f"CCS prediction input shape: {inputs.shape}")

    koina_model = Koina(model, "koina.wilhelmlab.org:443")
    try:
        predictions = koina_model.predict(inputs, debug=True)
    except Exception as e:
        print(f"CCS prediction error: {e}")
        raise

    predictions["non_switched"] = switched

    return predictions


def obtain_predictions_pairs(
    peptides,
    charges=[2],
    collision_energies=28,
    instrument_types="LUMOS",
    fragmentation_types="HCD",
    switched=False,
    model="UniSpec",
):
    """
    Function to obtain intensity predictions for a set of peptides.
    """
    num_peptides = peptides.shape[0]

    inputs = pd.DataFrame()
    inputs["peptide_sequences"] = np.array(peptides)
    inputs["precursor_charges"] = np.array(charges)
    # inputs["precursor_charges"] = np.array(num_peptides * charges)
    if collision_energies is not None:
        inputs["collision_energies"] = np.array(num_peptides * [collision_energies])
    inputs["instrument_types"] = np.array(num_peptides * [instrument_types])
    inputs["fragmentation_types"] = np.array(num_peptides * [fragmentation_types])
    inputs.drop_duplicates(inplace=True)
    print(inputs)

    model = Koina(model, "koina.wilhelmlab.org:443")
    try:
        predictions = model.predict(inputs, debug=True)
    except Exception as e:
        print(model.response_dict)
        print(model.response_dict())
        input()

    predictions["annotation"] = predictions["annotation"].map(
        lambda x: x.decode("utf-8")
    )

    predictions["non_switched"] = switched

    return predictions
