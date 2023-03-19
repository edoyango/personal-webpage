---
title: Setting Up Raspberry Pi Slurm Cluster
weight: 1
booktoc: false
---

# Setting Up Raspberry Pi Slurm Cluster


1. Flash disk(s) with raspberry pi lite. Insert disk(s) into pi(s) and power them on.
2. Run `sudo raspi-config`
    * update raspi-config
    * change hostname
    * setup ssh
    * change password for pi user
    * set wlan locale
    * set timezone
3. setup login-less ssh between nodes.
    * create key on one of the nodes

            sudo ssh-keygen # save it somewhere central like in /etc/ssh

    * configure ssh to use the newly created key by editing `/etc/ssh/ssh_config` and adding

           IdentityFile /etc/ssh/<key name>

    * make the key readable

            chmod +r /etc/ssh/<key name>

    * for each other node, copy ssh key to the nodes (e.g. with `scp`) and repeat above steps.
4. update `/etc/hosts` with all hosts names. e.g.:

        10.0.0.1 pimananager
        10.0.0.2 picompute1
        10.0.0.3 picompute2

5. Setup dhcp server on manager/login node (detailed instructions)
    * Install packages required for this step:

            sudo apt install iptables dnsmasq

    * In `/etc/dhcpcd.conf`, add the following:

            interface eth0 # for the ethernet network
            static ip_address=10.0.0.1/8 # provide static ip of 10.0.0.1
            static domain_name_servers=8.8.8.8,8.8.4.4 #???
            nolink # sets up the interface without being attached to ethernet

            move default config file and create new one with text below:

            sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.old
            sudo nano /etc/dnsmasq.conf

            interface=eth0
            listen-address=10.0.0.1
            dhcp-range=10.0.0.32,10.0.0.128,12h
            dhcp-host=<mac-address of nodes>,10.0.0.2
            dhcp-host=<mac-address of nodes>,10.0.0.3
            ...
            server=8.8.8.8
            server=8.8.4.4
            bind-interfaces
            domain-needed
            bogus-priv
            expand-hosts

    * add `sleep 10` to very start of `/etc/init.d/dnsmasq`. **needs to be improved on**
    * reboot manager/login node
    * from `/etc/sysctl.conf`, uncomment `net.ipv4.ip_forward=1`
    * Add iptables rules:

            sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
            sudo iptables -A FORWARD -i wlan0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
            sudo iptables -A FORWARD -i eth0 -o wlan0 -j ACCEPT

    * ensure iptables rules are presistent between boots:

            sudo apt install iptables-persistent

    * reboot switch and check DHCP leases were granted:

            cat /var/lib/misc/dnsmasq.leases

    * Internet should be being forwarded to new nodes
6. update existing packages on all nodes:

        sudo apt update; sudo apt upgrade -y; sudo apt update

7. packages to install:

        sudo apt install cmake tcl tcl-dev ntpdate tmux git slurm-wlm openmpi-bin openmpi-common libopenmpi3 \
            libopenmpi-dev git tree -y

8. turn off swapfiles (kills the sd card):

        sudo dphys-swapfile swapoff
        sudo dphys-swapfile uninstall
        sudo update-rc.d dphys-swapfile remove

9. Setup NFS for shared filesystem:
    * on the node with the drive connected:
        * install nfs server package:

                sudo apt install nfs-kernel-server -y

        * identify partition with `lsblk` and corresponding UUID with `blkid`
        * format disk:

                sudo mkfs -t ext4 /dev/<partition>

        * make directory drive is to be mounted on:

                sudo mkdir <directory>

        * mount drive by `sudo nano /etc/fstab` add the following:

                UUID=<UUID> <directory to mount to> ext4 defaults 0 <next integer in the list>

        * export the nfs by editing `sudo nano /etc/exports/` and add the following

                <directory to export> <ip style>/<search format>(rw,sync,no_root_squash,no_subtree_check)

            examples:

                /clusterfs 192.168.1.0/255.255.255.0(rw,sync,no_root_squash,no_subtree_check)
                /clusterfs 10.0.0.0/8.8.8.0(rw,sync,no_root_squash,no_subtree_check)
            								

        * mount the drive and then export:

                sudo mount -a
                sudo exportfs -a

    * on the nodes not connect to the drive:
        * install nfs common package:

                sudo apt install nfs-common -y

        * make directory drive where nfs folder is to be located. For purposes of MPI programs, make the folder name the same on all nodes
        * mount drive by `sudo nano /etc/fstab` and add the following lines:

                <ip of node with drive>:<directory> <directory> nfs defaults 0 <next index>

        * mount drive:

                sudo mount -a

10. install spack on nfs and setup spack env to load for root

    * Clone the Spack repo

            cd /clusterfs
            git clone -c feature.manyFiles=true https://github.com/spack/spack.git

    * add spack setup script to root .bashrc by sudo nano /root/.bashrc and add:

            . /share/spack/setup-env.sh

        **(would prefer this to be setup system wide)**
