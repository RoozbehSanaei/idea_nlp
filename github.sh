repository_name=examples
git config --global credential.helper cache
git init
git add *
git commit -a -m "Moving it to github"
git remote add $repository_name https://github.com/RoozbehSanaei/examples.git
git remote -v
git push -u $repository_name master