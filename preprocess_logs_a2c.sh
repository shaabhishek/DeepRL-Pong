#usage: ./preprocess_logs.sh filename
# out files: filename_reward.csv, filename_qval.csv

sed 's/INFO:root://' $1 > temp
# head -n 10 temp > parameters_$1

sed -n "/Rew:/p" temp > tempreward
sed 's/://g; s/Iter//g; s/Time//g; s/Rew//g; s/EpLength//g' tempreward > $1_reward.csv
rm tempreward

# sed -n "/Epsilon:/p" temp > logtext_epsilon_$1
# sed 's/Step:\|Epsilon:\|\s\+//g' logtext_epsilon_$1 > logtext_epsilon_$1.csv
# rm logtext_epsilon_$1

sed -n "/PGLoss:/p" temp > tempqval
sed 's/://g; s/Iter//g; s/PGLoss//g; s/ValLoss//g; s/Entropy//g; s/ExplainedVar//g' tempqval > $1_losses.csv
rm tempqval

rm temp