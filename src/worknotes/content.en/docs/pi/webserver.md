---
title: Setting Up Public Webserver on Raspberry Pi
weight: 1
---

# Setting Up Public Webserver on Raspberry Pi

The instructions here assumes you're using Raspberry Pi Lite as the OS on the Raspberry Pi. Other OS' are largely similar though. The main difference will be the packages and package managers, and the firewall tool.

* Setup the pi to host the server
    1. Flash disk with raspberry pi lite. Insert disk into pi and power them on.
    2. (follow the instructions [here](https://linuxhint.com/install_apache_web_server_ubuntu/) up to step 4)
* Setup router to forward http/https/ssh requests to the Raspberry Pi
    1. Obtain MAC address of Raspberry Pi e.g.,

            $ ip addr
            1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
            	    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
            	    inet 127.0.0.1/8 scope host lo
            	       valid_lft forever preferred_lft forever
            	    inet6 ::1/128 scope host
            	       valid_lft forever preferred_lft forever
            2: eth0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
            	    link/ether b8:27:eb:24:cb:98 brd ff:ff:ff:ff:ff:ff
            	    inet 10.0.0.1/8 brd 10.255.255.255 scope global noprefixroute eth0
            	       valid_lft forever preferred_lft forever
            	    inet6 fe80::1209:6b01:5af1:d12b/64 scope link tentative
            	       valid_lft forever preferred_lft forever
            3: wlan0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
            	    link/ether b8:27:eb:71:9e:cd brd ff:ff:ff:ff:ff:ff
            	    inet 192.168.0.104/24 brd 192.168.0.255 scope global dynamic noprefixroute wlan0
            	       valid_lft 4773sec preferred_lft 3873sec
            	    inet6 fe80::9a0:b632:2b19:bb89/64 scope link
            	       valid_lft forever preferred_lft forever

        and the MAC address is the first address following `link/ether` for the interface you're using. If the Raspberry Pi is connected to the router via WiFi, then you use the `wlan` address. If connected via ethernet cable, then you use the `eth0` address. In the above example, I used the address `b8:27:eb:71:9e:cd`, because my Pi is connected to the router via Wifi. We use this MAC address to link the physical device to an IP address in the next step.
    2. On the router:
        1. Assign a static IP address to the Pi. Instructions are readily available online for most manufacturers. To do this on my routher (TP-Link AC1200), I browsed to `http://192.168.0.1`, signed in, and then went to "IP & MAC Binding" → "Binding Settings" → "Add New" and enter the MAC address collected from above and the desired IP to bind it to.
        2. Setup port forwarding to forward packages to port 22, 80, 443, and 8443 to the IP address you've assigned the pi in the previous step. On my router, I navigated to "Forwarding" → "Virtual Server" → "Add New". This step forward incoming requests matching those ports, to the Raspberry Pi.
        3. Test the setup. While on a machine connected to the router, you can test ssh and http connection from the command line with `ssh pi@<ip-address>` and `curl http://<ip-address>`.
        4. You can also check http by entering the URL into your web browser: `http://<ip-address>`. Note that `<ip-address>` is the static IP address you assigned to the Pi on the router. 
    3. Sign an SSL certificate to enable HTTPS connections. This requires signing an SSL certificate. There are many paid services out there, but the free [Let's Encrypt](https://letsencrypt.org/) SSL certificate signing service is the best option (because it's free). Using the automatic cert-bot option should be fine ([instructions for Ubuntu](https://certbot.eff.org/instructions?ws=apache&os=ubuntufocal)). You can test that the signed certificate works, by navigating to the Pi web server in a browser on another device i.e., `https://<ip-address>` (note the "s").
* Get your ISP to open ports 22, 80, 443, and 8443 to your house. I had to send an online request to them and they followed up with me on the phone. Some ISPs may not be blocking any ports either.
* Purchase a domain and setup [resource records](https://support.google.com/domains/answer/3290350) so that the domain points your IP address. I setup:

        Host Name	Type	TTL		Data
        ed-yang.com	A	10 minutes	<my IP>
        ed-yang.com	CAA	10 minutes	0 issues "letsencrypt.org"
        www.ed-yang.com	CNAME	10 minutes	ed-yang.com

# Setting up a development webserver on WSL2
If you're on a Windows machine like I am, you can use WSL2 to test the html code. To setup WSL2, follow the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install).

You then need to setup `systemd` so the Apache web server package will work properly. To set it up, follow the instructions below (reproduced from [these instructions](https://devblogs.microsoft.com/commandline/systemd-support-is-now-available-in-wsl/) in case the page disappears):

1. Inside the WSL2 instance, edit/create the `/etc/wsl.conf` file and add the following lines of code:

        [boot]
        systemd=true

2. Restart WSL by exiting WSL, opening a powershell terminal as administrator, and running the command `wsl --shutdown`. After doing so, `systemd` should be running!

You can now go ahead and install the Apache web server packages:

```
sudo apt install apache2
```

and then copy your html code into `/var/www/html`.