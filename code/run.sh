#!/usr/bin/env bash

# Shell script to execute all code required to reproduce results
echo -e "-----\n Running code repository \n-----\n"


echo -e "-----\n Compute PVC metrics and do regressions "
mkdir -p output
python compute_vv_stats.py
python compute_regressions.py

echo -e "-----\n Compute AIC scores for each model"
python compute_aic.py


echo -e "-----\n Make Figure 2 - Scatter plots of VV vs VN and NV vs NN"
cd figure2
python make_fig.py

echo -e "-----\n Make figure 3 - Stacked linear regressions"
cd ../figure3
python make_fig.py

echo -e "-----\n Make figure 4 - Histograms of slope of linear regression"
cd ../figure4
python make_fig.py

echo -e "-----\n Make figure 5 - AP simulations"
cd ../figure5
python make_fig.py

echo -e "-----\n Make figure 6 - Stacked linear regressions for model simulations"
cd ../figure6
python make_fig6_reentry.py
python make_fig6_reentry_cond.py
python make_fig6_ead.py

