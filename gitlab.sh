repository_name=wiki
git config --global credential.helper cache
git init
git add *
git commit -a -F progress.txt
git remote add $repository_name https://RoozbehSanaei@gitlab.com/RoozbehSanaei/wiki.git
git remote -v
git push -u $repository_name master