python preprocess_daily_ohlc.py \
  ../../../data/daily-us-stocks-etfs/ETFs/spy.us.txt \
  ../historical_data \
  -s 0.9

python run_dqn.py \
  --env_name Discrete-1-Equity-v0 \
  --ep_len 200 \
  --lookback_num 200 \
  --exp_name spy-dqn-lb200-ep200-lstm \
  -train ../historical_data/spy.us_train.csv \
  --save_sess_freq 10000 \
  -test ../historical_data/spy.us_test.csv \
  # --double_q


# python run_dqn.py \
#   --env_name Discrete-1-Equity-v0 \
#   --ep_len 250 \
#   --lookback_num 5 \
#   --exp_name spy-dqn-lb5-ep250-test-on-train \
#   -test ../historical_data/spy.us_train.csv \
#   --only_test \
#   -l ../data/dqn_spy-dqn-lb5-ep250_Discrete-1-Equity-v0_17-12-2019_07-14-47/sess_final.ckpt
