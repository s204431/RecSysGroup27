## Setup :wrench:
login to your DTU account 
```bash
ssh <student number>@student.hpc.dtu.dk
```

Create a ssh key
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

Copy the pubic key and insert it into GitHub
```bash
cat ~/.ssh/id_rsa.pub
```

Run the following command when you want to activate the enviroment
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

Add enviroment
```bash
python -m venv ~/venv/RecSys
```

Activate enviroment
```bash
source ~/venv/RecSys/bin/activate
```

Update your enviroment (remember to be in RecSysGroup27 folder)
```bash
pip install -r requirements.txt
```













