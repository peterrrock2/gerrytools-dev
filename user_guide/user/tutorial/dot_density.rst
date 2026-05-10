.. code:: ipython3

    import geopandas as gpd
    import pandas as pd
    from gerrytools.plotting.geometry.dotdensity import DotDensityPlot

.. code:: ipython3

    # from gerrytools.logging import configure_logging
    # _ = configure_logging(level = "DEBUG")


.. code:: ipython3

    base_gdf = gpd.read_parquet("../../dev_files/dot_density/48_bg_2020.parquet").set_index("GEOID20")
    bg_pops = pd.read_parquet("../../dev_files/dot_density/pop_data/48_bg_pop_cat.parquet")
    gdf = gpd.GeoDataFrame(base_gdf.join(bg_pops))
    gdf




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>STATEFP20</th>
          <th>COUNTYFP20</th>
          <th>TRACTCE20</th>
          <th>BLKGRPCE20</th>
          <th>NAMELSAD20</th>
          <th>MTFCC20</th>
          <th>FUNCSTAT20</th>
          <th>ALAND20</th>
          <th>AWATER20</th>
          <th>INTPTLAT20</th>
          <th>INTPTLON20</th>
          <th>geometry</th>
          <th>tot_pop_20</th>
          <th>bpop_20</th>
          <th>hpop_20</th>
          <th>asian_nhpi_pop_20</th>
          <th>amin_pop_20</th>
          <th>other_pop_20</th>
          <th>white_pop_20</th>
        </tr>
        <tr>
          <th>GEOID20</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>481539506003</th>
          <td>48</td>
          <td>153</td>
          <td>950600</td>
          <td>3</td>
          <td>Block Group 3</td>
          <td>G5030</td>
          <td>S</td>
          <td>9238641</td>
          <td>0</td>
          <td>+33.9697231</td>
          <td>-101.3632430</td>
          <td>POLYGON ((-101.37703 33.98424, -101.37701 33.9...</td>
          <td>736</td>
          <td>9</td>
          <td>310</td>
          <td>3</td>
          <td>5</td>
          <td>2</td>
          <td>407</td>
        </tr>
        <tr>
          <th>481539506005</th>
          <td>48</td>
          <td>153</td>
          <td>950600</td>
          <td>5</td>
          <td>Block Group 5</td>
          <td>G5030</td>
          <td>S</td>
          <td>1070676551</td>
          <td>101332</td>
          <td>+33.9417578</td>
          <td>-101.2985920</td>
          <td>POLYGON ((-101.56453 34.05969, -101.56393 34.0...</td>
          <td>430</td>
          <td>10</td>
          <td>115</td>
          <td>2</td>
          <td>0</td>
          <td>8</td>
          <td>295</td>
        </tr>
        <tr>
          <th>481610004001</th>
          <td>48</td>
          <td>161</td>
          <td>000400</td>
          <td>1</td>
          <td>Block Group 1</td>
          <td>G5030</td>
          <td>S</td>
          <td>12766486</td>
          <td>8204</td>
          <td>+31.7876907</td>
          <td>-096.4702707</td>
          <td>POLYGON ((-96.49643 31.79633, -96.49558 31.796...</td>
          <td>875</td>
          <td>137</td>
          <td>119</td>
          <td>3</td>
          <td>16</td>
          <td>8</td>
          <td>592</td>
        </tr>
        <tr>
          <th>481610006001</th>
          <td>48</td>
          <td>161</td>
          <td>000600</td>
          <td>1</td>
          <td>Block Group 1</td>
          <td>G5030</td>
          <td>S</td>
          <td>153883256</td>
          <td>544987</td>
          <td>+31.5680364</td>
          <td>-096.3135009</td>
          <td>POLYGON ((-96.42136 31.68625, -96.42085 31.686...</td>
          <td>721</td>
          <td>61</td>
          <td>103</td>
          <td>8</td>
          <td>14</td>
          <td>10</td>
          <td>525</td>
        </tr>
        <tr>
          <th>481759602001</th>
          <td>48</td>
          <td>175</td>
          <td>960200</td>
          <td>1</td>
          <td>Block Group 1</td>
          <td>G5030</td>
          <td>S</td>
          <td>391287006</td>
          <td>1789495</td>
          <td>+28.7110687</td>
          <td>-097.4683451</td>
          <td>POLYGON ((-97.65396 28.75391, -97.65384 28.754...</td>
          <td>1548</td>
          <td>50</td>
          <td>408</td>
          <td>14</td>
          <td>16</td>
          <td>12</td>
          <td>1048</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>482211602161</th>
          <td>48</td>
          <td>221</td>
          <td>160216</td>
          <td>1</td>
          <td>Block Group 1</td>
          <td>G5030</td>
          <td>S</td>
          <td>8063739</td>
          <td>2334801</td>
          <td>+32.5329731</td>
          <td>-097.8324117</td>
          <td>POLYGON ((-97.85025 32.50288, -97.84974 32.504...</td>
          <td>928</td>
          <td>17</td>
          <td>112</td>
          <td>3</td>
          <td>45</td>
          <td>9</td>
          <td>742</td>
        </tr>
        <tr>
          <th>482211603031</th>
          <td>48</td>
          <td>221</td>
          <td>160303</td>
          <td>1</td>
          <td>Block Group 1</td>
          <td>G5030</td>
          <td>S</td>
          <td>1542092</td>
          <td>0</td>
          <td>+32.4270925</td>
          <td>-097.8017578</td>
          <td>POLYGON ((-97.81509 32.42052, -97.81509 32.420...</td>
          <td>1105</td>
          <td>8</td>
          <td>132</td>
          <td>30</td>
          <td>24</td>
          <td>14</td>
          <td>897</td>
        </tr>
        <tr>
          <th>480459502002</th>
          <td>48</td>
          <td>045</td>
          <td>950200</td>
          <td>2</td>
          <td>Block Group 2</td>
          <td>G5030</td>
          <td>S</td>
          <td>2320609023</td>
          <td>4068657</td>
          <td>+34.5251725</td>
          <td>-101.2058930</td>
          <td>POLYGON ((-101.47208 34.44874, -101.47203 34.4...</td>
          <td>791</td>
          <td>18</td>
          <td>151</td>
          <td>0</td>
          <td>14</td>
          <td>2</td>
          <td>606</td>
        </tr>
        <tr>
          <th>482870001004</th>
          <td>48</td>
          <td>287</td>
          <td>000100</td>
          <td>4</td>
          <td>Block Group 4</td>
          <td>G5030</td>
          <td>S</td>
          <td>104544628</td>
          <td>1415220</td>
          <td>+30.3676576</td>
          <td>-097.1392129</td>
          <td>POLYGON ((-97.26301 30.36836, -97.26182 30.370...</td>
          <td>1091</td>
          <td>35</td>
          <td>148</td>
          <td>9</td>
          <td>28</td>
          <td>7</td>
          <td>864</td>
        </tr>
        <tr>
          <th>482870001002</th>
          <td>48</td>
          <td>287</td>
          <td>000100</td>
          <td>2</td>
          <td>Block Group 2</td>
          <td>G5030</td>
          <td>S</td>
          <td>266509965</td>
          <td>874440</td>
          <td>+30.4814991</td>
          <td>-097.0140837</td>
          <td>POLYGON ((-97.13472 30.46814, -97.13418 30.468...</td>
          <td>1314</td>
          <td>54</td>
          <td>106</td>
          <td>2</td>
          <td>25</td>
          <td>23</td>
          <td>1104</td>
        </tr>
      </tbody>
    </table>
    <p>18638 rows × 19 columns</p>
    </div>



.. code:: ipython3

    dplot = DotDensityPlot(
        gdf,
        outline_column="COUNTYFP20",
        show_labels=False,
        people_per_dot=100,
        dpi=2000
    )

.. code:: ipython3

    daves_colors = {
        'White': "#d5cbdd",
        'Asian': "#db2f28",
        'Black': "#ffc50c",
        'Latino': "#91b321",
    }
    
    dplot.set_marker_options(markersize=0.1)
    
    dplot.add_dot_density(
        column_name="tot_pop_20",
        color=daves_colors["White"],
        n_chunks=100,
        force_new_dots=True,
    )
    dplot.add_dot_density(
        column_name="hpop_20",
        color="#ffc50c",
        n_chunks=100,
        force_new_dots=True,
    )
    dplot.add_dot_density(
        column_name="asian_nhpi_pop_20",
        color="#db2f28",
        force_new_dots=True,
    )
    # dplot.focus_axes(
    #     geometry_mask=(gdf["COUNTYFP20"]=='325')
    # )
    dplot.show()


.. parsed-literal::

    Generating dots for column 'tot_pop_20'.
    Generating dots for column 'hpop_20'.
    Generating dots for column 'asian_nhpi_pop_20'.
    Rendering 1 outline layer...
    Rendering 420,214 dots for columns '['tot_pop_20', 'hpop_20', 'asian_nhpi_pop_20']'...



.. image:: dot_density_files/dot_density_5_1.png


.. code:: ipython3

    dplot.save_legend("dot_legend.png")

