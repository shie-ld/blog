
# 7 Essential Linux Commands for Managing Users

Linux user admin commands


From the very beginning, the Linux operating system was designed to be a multi-user OS. As such, one of the most common administrative tasks performed on a Linux machine is managing user accounts. It’s a critical part of keeping a healthy and secure Linux machine.

You might think that it is overwhelming to manage users from the command line. On the contrary, it is not at all. There are only a few basic commands that you need to know, and I will cover them in this article.

## Background

A user account in Linux is the primary way of gaining access to a system — whether locally or remotely.

There are three main types of user accounts on a Linux system:

1. Root account — User with unlimited access to modify the Linux system.

1. System accounts — Used for running services or specific programs. Some of the most common ones include MySQL, mail, daemon, bin, etc.

1. User accounts — General users who have limited access to the system.

In Linux, most user information can be found in three files located at /etc/passwd, /etc/shadow, and /etc/group.

*Note: You will need elevated privileges to run some of the commands specified in this guide.*

## 1. List All Users

To list all available users on a machine, you can run the following command:

    $ compgen -u

Alternatively, you can output the users straight from the /etc/passwd file using the following command:

    $ cat /etc/passwd

As you notice from the output, your list will contain the root user, several system users, and general user accounts. The output will be similar to this:

![](https://cdn-images-1.medium.com/max/2366/1*djLQlx3RFU4hEdkRurBN-A.png)

## 2. Create a User Account

One of the most common administrative tasks is adding users to a system. The simple command for that is useradd. For example, to add a user named Marion, we can run the following command:

    $ sudo useradd -c "audit consultant" marion

The -c is an optional argument that allows you to add a comment associated with the user you are creating.

The useradd command takes other optional arguments too. You can take a look at them by running the following command:

    man useradd

*Tip:** **Debian-based systems have the adduser command as an alternative.*

## 3. Change a User Password

To add a default password to the user we just created above, we can use the passwd command. The passwd command can also be used to modify the password of any user as follows:

    $ sudo passwd marion

You will then be prompted to enter the password you wish to set. The output will be similar to this:

![](https://cdn-images-1.medium.com/max/2180/1*ujblOz6tgxlhYSCF6f945g.png)

*Note: Nothing will be displayed on the terminal as you type the password. Let’s just say this is the Unix way of doing things.*

## 4. Switching User Accounts

As I mentioned earlier, Linux is truly a multi-user OS. You can switch user accounts from the terminal as you wish (as long as you are allowed to do so).

To switch to the marion user account that we just created above, we can use the su command as follows:

    $ su marion

You will then be prompted to enter the password of the user you are switching to.

If you successfully switched users, you can confirm your new identity by running the following command:

    $ whoami

Your output should be similar to the one below:

![](https://cdn-images-1.medium.com/max/2000/1*4TozUDCVZ7xtAIihJIQtMw.png)

To switch back to the previous account, just type the exit command.

## 5. Modifying a User Account

The usermod command allows you to make changes to user accounts. It takes similar optional arguments as the useradd command.

For example, to modify the comment for the marion user account that we created above, you can do the following:

    $ sudo usermod -c "New audit consulant comment" marion

To check if the comment was indeed modified, we can search for the user account name in the /etc/passwd file using the following command:

    $ grep 'marion' /etc/passwd

![](https://cdn-images-1.medium.com/max/2606/1*Fjz_qu0_wBBxpL7kdzzvRw.png)

## 6. Delete a User Account

Deleting a user account from the command line is extremely easy. Therefore, you need to practice caution.

The userdel command is used to delete a user account. It only takes a single optional argument: -r. When the -r argument is specified, you delete the user’s home directory and the mail spool.

To delete the user account that we created in this guide, do the following:

    $ sudo userdel -r marion

## 7. Running Commands as a Superuser

We have already used the command we will look at now, but I didn’t really explain it. The sudo (superuser do) command allows you to run commands as the root user. You will be prompted to enter your user password to run this command.

If you can’t run commands as the root user, you will be notified via the terminal that you cannot run sudo on the system.

To run elevated commands, use the sudo command followed by the elevated command you want to run with superuser privileges. For example, to add a user, you can do this:

    $ sudo useradd newuser

## Final Thoughts

There you have it: a few powerful commands that will allow you to administer user accounts on a Linux system.

