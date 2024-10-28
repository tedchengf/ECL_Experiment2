<hr>

## Before the experiment

- Send subjects a reminder 10-20 minutes in advance
- Print all **Consent Forms** and **Debriefing Forms**
- Set up the experiment on the lab computers
  - You first need to fill in the google sheet about the subjects' condition and ID
  - Then on the computer, check that your conda environment is `Experiment`. If it's not, enter `conda activate Experiment`. Then head to the experiment folder with `cd` and then `cd Desktop/ECL_Experiment2`. Use `pwd` to make sure you are in the right directory.
  - Now enter `python3 T2_L1.py` for list 1 and `python3 T2_L2.py` for list 2.
  - Once the program starts, you first enter the subject ID and then the rule condition (`T1, T2, B1, B2, S1`). Double check you are on the right list and the right rule condition, this is **VERY IMPORTANT**.
    \## When the subjects arrive
- Check the studies the subjects are participating in (usually not necessary)
- If the subjects are late, make sure they have at least $40$ minutes to complete the task. Otherwise, tell them that they are too late and kindly ask them to leave (and no credit will be granted).
- Briefly introduce yourself
- Let the subject sign the **Consent Form**. When you receive the form, check the following details:
  - All 4 blanks needed to be filled (*age*, *signature*, *date*, and *name in print*)
  - Check the date
  - There cannot be any cross-out texts or corrections. This usually happens when subjects put in the wrong date. If you see cross-out texts or corrections, make them sign a new form.
- If everything is fine, you can collect the consent form and lead subject into the experiment room
  \## When the subjects are introduced to the experiment room
- You should lead them to the room and say something resembling the following line
  \> You can start the experiment now and follow the on-screen instructions. However, you should stop when you reach the practice block -- which will be around 12 slides after this slide. I will find you then an explain a few more things.
- Before you leave, check if the subjects have any questions.
- *Note: It usually takes subjects 3-5 minutes to reach the practice block, so you can use the time to admit another subject into the room*
  \## When the subjects reach the practice block
- Knock to go into the room
- You should find the subject waiting for you before the practice block or in the middle of the practice block. If they already proceed into the training block, you can still demonstrate and explain the experiment.
- First, explain that they can make a prediction by pressing `T` and `F`, where `T` means the objects will explode, and `F` means the objects will not explode.
- Once they make the prediction, demonstrate that they can change their prediction by hitting a different key (`T` and `F`). You should go in and type the two keys into the keyboard while making the following explanation:
  \> Note that you can change your predictions ***(switch `T` and `F` on the keyboard)***. Only the last one will be recorded. However, once you press proceed you may see that the text here ***(point to the True and False text)*** will flicker a little bit. It's only a presentation bug, pay no attention to it.
- Next, ask them to advance to the confidence rating slide. You should explain the meaning of the rating, and demonstrate how they can redo the trial through the following explanation:
  \> This is the confidence rating scale, where $5$ means you are $100\%$ confident and $1$ means you are just guessing, which you probably are right now. $0$ here ***(press 0 on keyboard)*** is reserved for situations where, you see the stimuli, made the prediction, and headed to this screen and realized that you've probably made the wrong decision. In those cases, you can always hit $0$ to go back and remake your prediction ***(press `Enter` to go back to the stimuli)***. Does that make sense?
- Let them finish the trial and see that they have no question about the trial structure.
- Now, you need to explain the task with the following explanations:
  \> So before you start, there are a few more things I need to reiterate. First, the experiment has three blocks, with the first two being the training blocks and the third one the testing block. Your task is to infer a rule that determines whether the combination will explode or not. The rule is **deterministic**, it **will not change throughout the entire experiment**, and **does not care about the order of the objects**. This rule is challenging but very much possible to solve, so I'd say you should pay close attention to the composition of the objects. That is, what are their shapes, their sizes, and their colors. Do you have any questions?
  \>
  \> Ok. So you can start now, but you should not take any notes. You also should not attempt to rely solely on memorization; this is not a memorization task. Do not press `Q` during the experiment (it's the force abort button), and try not to kick the cables beneath the cables, all right?
- If they have no questions, then say the following before exiting the room:
  \> All right, you are all good to start. You can find me outside if you have any questions or when you are done with the experiment. Good luck!
  \## When the subjects finish the experiment
- Lead them outside and let them read through and sign the **Debriefing Form**.
- Ask them about the rules they've learned and record them.
  - Pay very close attention to **how they describe the rules** (for example, some subjects say something like the triangle is the ignition and the circle is the fuel), and also what rules they've tried. This will give us critical insight into what are the rules subjects naturalistically come up with during the experiment.
- Explain the ground truth rule to them, and answer their questions.
- Finally, keep the **Debriefing Form** and give them a copy of the **Consent Form**, and conclude the experiment
  \## When the subjects terminate the experiment prematurely
- If the subjects have any emergency, help them and terminate the experiment. Grant them credit for showing up and starting the experiment.
- If the subject accidentally ended the experiment (program failure / pressing `Q` / kicking the power cable so the computer looses power):
  - If there are at least $40$ minutes left **and** if the subject weas only around $20$ trials into the first block, you can restart the experiment under the same condition (remember to delete the subjects' response file in the directory `Desktop\ECL_Experiment2\rsps\` before restarting the experiment).
  - Otherwise, tell them that it's all right and that you will grant them credit nonetheless, and conclude the experiment.
- Note this incidence in the log file on the Google sheet, and delete the subject's response file in the directory `Desktop\ECL_Experiment2\rsps\`. Forfeit their **Consent Form** as well.

## Wrapping Up

- Staple the **Consent Form** and the **Debriefing Form** together. Make sure the names match. You can leave them onto my desk, or bring it with you and give it to me the next time.
- Check the computer files and make sure that the subjects' data are recorded in `Desktop\ECL_Experiment2\rsps\`. If it's not, do not use that computer and let me know.
- If you've set up a computer but the subject do not show up, you have two options:
  - If you have another session later, you can simply leave the computer as is and let the next subject use it
  - If this is the last session, then press `Q` to quite the program. Delete the subject's data in `Desktop\ECL_Experiment2\rsps\` (which will be created as soon as you enter the subject ID into the program).
    \## Extra Notes
- If the participants are in a hurry, you can assign them `T1` or `S1`. The two rule conditions usually takes less than $30$ minutes to complete.
- Make sure you do a full cycle within 1 list before starting the next. That is, you should only switch list after completing all the 5 rule conditions.
