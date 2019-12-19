python preprocess_daily_ohlc.py \
  ../../../data/daily-us-stocks-etfs/ETFs/spy.us.txt \
  -f2 ../../../data/daily-us-stocks-etfs/ETFs/gld.us.txt \
  ../historical_data/ \
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
        --env_name Discrete-2-Equities-v0 \
        --ep_len $ep_len \
        --lookback_num $lookback_num \
        --model $model \
        --exp_name "report-spy-gld-price_variance-lb$lookback_num-ep$ep_len-$model" \
        -train ../historical_data/spy.us_pair_train.csv \
        -train2 ../historical_data/gld.us_pair_train.csv \
        --save_sess_freq 5000 \
        -test ../historical_data/spy.us_pair_test.csv \
        -test2 ../historical_data/gld.us_pair_test.csv
    done
  done
done


# python run_dqn.py \
#   --env_name Discrete-2-Equities-v0 \
#   --ep_len 200 \
#   --lookback_num 200 \
#   --exp_name spy-gld-dqn-lb200-ep200 \
#   -train ../historical_data/spy.us_pair_train.csv \
#   -train2 ../historical_data/gld.us_pair_train.csv \
#   --save_sess_freq 10000 \
#   -test ../historical_data/spy.us_pair_test.csv \
#   -test2 ../historical_data/gld.us_pair_test.csv \
#   # --double_q



# python run_dqn.py \
#   --env_name Discrete-1-Equity-v0 \
#   --ep_len 2 \
#   --double_q \
#   --lookback_num 20 \
#   --exp_name spy-dqn-lb20-ep1-test-on-train \
#   -test ../historical_data/spy.us_train.csv \
#   --only_test \
#   -l ../data/dqn_double_q_spy-dqn-lb20-ep1_Discrete-1-Equity-v0_17-12-2019_02-34-01/sess_final.ckpt
