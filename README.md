<div style="display: flex; align-items: center;">
  <img src="images/HydrakonNoBGOrche.png" alt="Hydrakon Logo" width="100" />
</div>

# Hydrakon | HWUD
**Hydrakon** is the central codebase for our participation in Formula Student UK 2025. It houses all the software systems used to control and operate the ADS-DV (Autonomous Driving System – Development Vehicle).
- Nvidia Jetson AGX Orin (64 GB)
- ZED2i Stereo Camera
- Robosense Helios 16
- CHCNAV CGI-410 (INS)
- TP-Link AX1500 Wi-Fi 6

## Network Settings
- **IP Configuration**
  - Jetson AGX Orin ➔ 192.168.1.100 (LAN1)
  - Robosense Helios 16 ➔ 192.168.1.200 (LAN2)
  - CHCNAV CGI-410 INS ➔ 192.168.1.201 (LAN3)

- **LiDAR configuration**
  - **Device IP Address** ➔ 192.168.1.200
  - **Device IP Mask** ➔ 255.255.255.0
  - **Device IP Gateway** ➔ 192.168.1.1
  - **Destination IP Address** ➔ 192.168.1.100 **(Jetson)**
  - **MSOP Port** ➔ 6699
  - **DIFOP Port** ➔ 7788
  - **Return Mode** ➔ Strongest
  - **Rotation Speed** ➔ Set it to 600/1200 as needed
  - **Operation Mode** ➔ High Performance
