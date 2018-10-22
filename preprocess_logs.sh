#usage: ./preprocess_logs.sh filename
# out files: filename_reward.csv, filename_qval.csv

sed 's/INFO:root://' $1 > temp
# head -n 10 temp > parameters_$1

sed -n "/Reward:/p" temp > tempreward
sed 's/://g; s/Episode//g; s/Reward//g; s/Length//g' tempreward > $1_reward.csv
rm tempreward

# sed -n "/Epsilon:/p" temp > logtext_epsilon_$1
# sed 's/Step:\|Epsilon:\|\s\+//g' logtext_epsilon_$1 > logtext_epsilon_$1.csv
# rm logtext_epsilon_$1

sed -n "/QValue:/p" temp > tempqval
sed 's/://g; s/Step//g; s/QValue//g; s/Loss//g' tempqval > $1_qval.csv
rm tempqval

rm temp