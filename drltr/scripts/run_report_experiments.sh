python preprocess_daily_ohlc.py \
  ../../../data/daily-us-stocks-etfs/ETFs/spy.us.txt \
  ../historical_data \
  -s 0.9

EP_LENS=(5 20 60)
LOOKBACK_NUMS=(5 20 60)
MODELS=('fc' 'lstm')

for ep_len in "${EP_LENS[@]}";
do
  for lookback_num in "${LOOKBACK_NUMS[@]}";
  do
    for model in "${MODELS[@]}";
    do
      python run_dqn.py \
        --env_name Discrete-1-Equity-v0 \
        --ep_len $ep_len \
        --lookback_num $lookback_num \
        --model $model \
        --exp_name "report-spy-price_variance-lb$lookback_num-ep$ep_len-$model" \
        -train ../historical_data/spy.us_train.csv \
        --save_sess_freq 5000 \
        -test ../historical_data/spy.us_test.csv \
        --double_q
    done
  done
done
