

import numpy as np
import pandas as pd


class Matrix(np.ndarray):
    """ defines a matrix class based off of numpy arrays to convert everything
    to standard mathematical formats etc

    Includes sample and feature dataframes for metadata storage, motivated by anndata package
    Author: John Lee 
    Oct 2024
    """

    def __new__(cls, input_array,  samples=None, features=None):
        # Step 1: Create a NumPy array (or subclass) from the input array
        input_array = np.asarray(input_array)

        if input_array.ndim == 1:
            input_array = input_array[:, np.newaxis]
        elif input_array.ndim == 0:
            input_array = input_array[np.newaxis, np.newaxis]
        elif input_array.ndim > 2:
            raise ValueError("Input data must be 1D, 2D")

        obj = np.asarray(input_array).view(cls)  # cls refers to MyArray
        n, m = obj.shape
        if samples is not None:
            assert len(
                samples) == n, f"len(samples) == {len(samples)}. Expected {n} "
        if features is not None:
            assert len(
                features) == m, f"len(features) == {len(features)}. Expected {m} "

        if isinstance(samples, pd.DataFrame):
            obj.samples = samples
        else:
            samps = samples if samples is not None else list(range(n))
            obj.samples = pd.DataFrame(index=samps)

        if isinstance(features, pd.DataFrame):
            obj.features = features
        else:
            feats = features if features is not None else list(range(m))
            obj.features = pd.DataFrame(index=feats)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.samples = getattr(
            obj, 'samples', None)
        self.features = getattr(
            obj, 'features', None)

    def __getitem__(self, key):
        """slices samples and features accordingly whe input matrix is sliced 

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        result = super().__getitem__(key)

        if isinstance(result, Matrix):
            if isinstance(key, tuple):
                row_slice, col_slice = key
                result.samples = self.samples.iloc[row_slice, :]
                result.features = self.features.iloc[col_slice, :]

            elif isinstance(key, slice):
                result.samples = self.samples.iloc[key, :]

        return result

    def __repr__(self):
        max_display = 3  # Number of rows/columns to display at the start and end
        max_decimal = 5
        ellipses = '...'
        # Truncate the index and columns for display
        index_display = list(self.samples.index)[:max_display] + [ellipses] + \
            list(self.samples.index)[-max_display:] if len(self.samples.index) > 2 * \
            max_display else list(self.samples.index)
        columns_display = list(self.features.index)[:max_display] + [ellipses] + \
            list(self.features.index)[-max_display:] if len(
            self.features.index) > 2 * max_display else list(self.features.index)

        # Prepare column headers
        col_header = " " * 12 + \
            " ".join([f"{col:>10}" for col in columns_display])
        col_header += '\n' + " " * 12 + \
            "".join(['_'*11 for _ in columns_display])
        # Prepare row data
        rows = []
        for i in range(min(max_display, self.shape[0])):
            if self.shape[1] <= 2*max_display:
                row_data = " ".join(
                    [f"{round(self[i, j],max_decimal):>10}" for j in range(min(2*max_display, self.shape[1]))])
            else:
                row_data = " ".join(
                    [f"{round(self[i, j],max_decimal):>10}" for j in range(min(max_display, self.shape[1]))])
                if self.shape[1] > 2 * max_display:
                    row_data += f"{ellipses:>10}" + \
                        " ".join(
                            [f"{round(self[i, j],max_decimal):>10}" for j in range(-max_display, 0)])

            rows.append(f"{index_display[i]:>10} | {row_data}")

        # Include ellipses for skipped rows
        if self.shape[0] > 2 * max_display:
            rows.append(ellipses)

        # Last rows
        for i in range(-max_display, 0):
            # Ensure the index is within the valid range
            # Only proceed if i is within the number of rows

            if abs(i) < self.shape[0] and ((self.shape[0] + i) >= max_display):

                if self.shape[1] <= 2*max_display:
                    row_data = " ".join(
                        [f"{round(self[i, j],max_decimal):>10}" for j in range(min(2*max_display, self.shape[1]))])

                else:
                    row_data = " ".join(
                        [f"{round(self[i, j], 5):>10}" for j in range(
                            min(max_display, self.shape[1]))]
                    )
                    if self.shape[1] > 2 * max_display:
                        row_data += f"{ellipses:>10}" + \
                            " ".join(
                                [f"{round(self[i, j], 5):>10}" for j in range(-max_display, 0)]
                            )

                rows.append(f"{index_display[i]:>10} | {row_data}")

        # Combine everything into the final representation
        return col_header + "\n" + "\n".join(rows)
