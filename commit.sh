git add .
message=""
git commit -m "$message"
git push

# randomText="origin`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 6`"
# git remote add ${randomText} https://github.com/cagriozcaglar/DNNapplicationsWithPyTorch.git
# git remote -v
# git push ${randomText} master