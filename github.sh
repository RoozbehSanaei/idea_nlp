repository_name=idea_nlp
git config --global credential.helper cache
git init
git add *
git commit -a -m "Moving it to github"
git remote add $repository_name https://github.com/RoozbehSanaei/idea_nlp.git
git remote -v
git push -u $repository_name master