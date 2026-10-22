set -u
cd /l/ladovj1/defumat
flock /l/ladovj1/git.lock git fetch -q /l/ladovj1/bundles/incompat-ultraseed.bundle +incompat/ultraseed:incompat-ultraseed
git rev-parse incompat-ultraseed
ls -d /l/ladovj1/w-inc-ultraseed* 2>&1
WT=/l/ladovj1/w-inc-ultraseed
if [ ! -e "$WT" ]; then
  git worktree add --detach "$WT" incompat-ultraseed
fi
git -C "$WT" rev-parse HEAD
mkdir -p /l/ladovj1/review/incompat/ultraseed
ls /l/ladovj1/review/incompat/ultraseed
pgrep -af slot.sh | wc -l
