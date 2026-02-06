/* SPDX-License-Identifier: BSD-2-Clause-Patent
 *
 * SPDX-FileCopyrightText: 2016-2020 the prplMesh contributors (see AUTHORS.md)
 *
 * This code is subject to the terms of the BSD+Patent license.
 * See LICENSE file for more details.
 */

#include "son_actions.h"

#include "controller.h"
#include "db/db_algo.h"
#include "tasks/agent_monitoring_task.h"
#include "tasks/association_handling_task.h"
#include "tasks/bml_task.h"
#include "tasks/btm_request_task.h"
#include "tasks/client_steering_task.h"

#include <bcl/network/network_utils.h>
#include <bcl/network/sockets.h>
#include <bcl/son/son_wireless_utils.h>
#include <easylogging++.h>

#include <beerocks/tlvf/beerocks_message_cli.h>
#include <tlvf/ieee_1905_1/tlvAlMacAddress.h>
#include <tlvf/ieee_1905_1/tlvSupportedFreqBand.h>
#include <tlvf/ieee_1905_1/tlvSupportedRole.h>
#include <tlvf/wfa_map/tlvAgentApMldConfiguration.h>
#include <tlvf/wfa_map/tlvClientAssociationControlRequest.h>
#include <tlvf/wfa_map/tlvProfile2MultiApProfile.h>

using namespace beerocks;
using namespace net;
using namespace son;

void son_actions::handle_completed_connection(db &database, ieee1905_1::CmduMessageTx &cmdu_tx,
                                              task_pool &tasks, std::string client_mac)
{
    LOG(INFO) << "handle_completed_connection client_mac=" << client_mac;
    std::shared_ptr<Station> client = database.get_station(tlvf::mac_from_string(client_mac));
    if (!client) {
        return;
    }
    if (!database.set_sta_state(client_mac, beerocks::STATE_CONNECTED)) {
        LOG(ERROR) << "set sta state failed";
    }
    // update bml listeners
    LOG(DEBUG) << "BML, sending connect CONNECTION_CHANGE for mac " << client_mac;
    auto new_event = std::make_shared<bml_task::connection_change_event>();
    new_event->mac = client_mac;
    tasks.push_event(database.get_bml_task_id(), bml_task::CONNECTION_CHANGE, new_event);

    auto new_hostap_mac      = database.get_sta_parent(client_mac);
    auto previous_hostap_mac = client->get_previous_bss()
                                   ? tlvf::mac_to_string(client->get_previous_bss()->bssid)
                                   : network_utils::ZERO_MAC_STRING;
    auto hostaps             = database.get_active_radios(); // snapshot (by value)

    hostaps.erase(new_hostap_mac); //next operations will be done only on the other APs

    if (database.is_sta_wireless(client_mac)) {
        LOG(DEBUG) << "node " << client_mac << " is wireless";
        /*
         * send disassociate request to previous hostap to clear STA mac from its list
         */
        if ((!previous_hostap_mac.empty()) &&
            (previous_hostap_mac != network_utils::ZERO_MAC_STRING) &&
            (previous_hostap_mac != new_hostap_mac)) {
            disconnect_client(database, cmdu_tx, client_mac, previous_hostap_mac,
                              eDisconnect_Type_Disassoc, 0);
        }

        /*
         * launch association handling task for async actions
         * and further handling of the new connection
         */
        auto new_task =
            std::make_shared<association_handling_task>(database, cmdu_tx, tasks, client_mac);
        tasks.add_task(new_task);
    }
}

bool son_actions::add_station_to_default_location(db &database, std::string client_mac)
{
    sMacAddr gw_lan_switch = network_utils::ZERO_MAC;

    auto gw = database.get_gw();
    if (!gw) {
        LOG(ERROR) << "add_station_to_default_location - can't get GW node";
        return false;
    }

    if (gw->eth_switches.empty()) {
        LOG(ERROR) << "add_station_to_default_location - GW has no LAN SWITCH node!";
        return false;
    }
    gw_lan_switch = gw->eth_switches.begin()->first;

    if (!database.add_station(network_utils::ZERO_MAC, tlvf::mac_from_string(client_mac),
                              gw_lan_switch)) {
        LOG(ERROR) << "add_station_to_default_location - add_station failed";
        return false;
    }

    if (!database.set_sta_state(client_mac, beerocks::STATE_CONNECTING)) {
        LOG(ERROR) << "add_station_to_default_location - set_sta_state failed.";
        return false;
    }

    return true;
}

void son_actions::unblock_sta(db &database, ieee1905_1::CmduMessageTx &cmdu_tx, std::string sta_mac)
{
    LOG(DEBUG) << "unblocking " << sta_mac << " from network";

    auto hostaps              = database.get_active_radios();
    const auto &current_bssid = database.get_sta_parent(sta_mac);
    const auto &ssid          = database.get_bss_ssid(tlvf::mac_from_string(current_bssid));

    std::unordered_set<sMacAddr> unblock_list{tlvf::mac_from_string(sta_mac)};

    for (const auto &hostap : hostaps) {
        /*
         * unblock client from all hostaps to prevent it from getting locked out
         */
        std::shared_ptr<Agent::sRadio> radio =
            database.get_radio_by_uid(tlvf::mac_from_string(hostap));
        if (!radio) {
            continue;
        }

        for (const auto &bss : radio->bsses) {
            if (!bss.second) {
                continue;
            }
            if (bss.second->ssid != ssid) {
                continue;
            }
            std::shared_ptr<Agent> agent = database.get_agent_by_radio_uid(radio->radio_uid);
            if (!agent) {
                continue;
            }

            son_actions::send_client_association_control(
                database, cmdu_tx, agent->al_mac, bss.second->bssid, unblock_list, 0,
                wfa_map::tlvClientAssociationControlRequest::UNBLOCK);
        }
    }
}

int son_actions::steer_sta(db &database, ieee1905_1::CmduMessageTx &cmdu_tx, task_pool &tasks,
                           std::string sta_mac, std::string chosen_hostap,
                           const std::string &triggered_by, const std::string &steering_type,
                           bool disassoc_imminent, int disassoc_timer_ms, bool steer_restricted)
{
    auto new_task = std::make_shared<client_steering_task>(
        database, cmdu_tx, tasks, sta_mac, chosen_hostap, triggered_by, steering_type,
        disassoc_imminent, disassoc_timer_ms, steer_restricted);

    tasks.add_task(new_task);
    return new_task->id;
}

int son_actions::start_btm_request_task(
    db &database, ieee1905_1::CmduMessageTx &cmdu_tx, task_pool &tasks,
    const bool &disassoc_imminent, const int &disassoc_timer_ms, const int &bss_term_duration_min,
    const int &validity_interval_ms, const int &steering_timer_ms, const std::string &sta_mac,
    const std::string &target_bssid, const std::string &event_source)
{

    auto new_task = std::make_shared<btm_request_task>(
        database, cmdu_tx, tasks, sta_mac, target_bssid, event_source, disassoc_imminent,
        validity_interval_ms, steering_timer_ms, disassoc_timer_ms);

    tasks.add_task(new_task);
    return new_task->id;
}

bool son_actions::set_radio_active(db &database, task_pool &tasks, std::string hostap_mac,
                                   const bool active)
{
    bool result = database.set_radio_active(tlvf::mac_from_string(hostap_mac), active);

    if (result) {
        auto new_event = std::make_shared<bml_task::connection_change_event>();
        new_event->mac = hostap_mac;
        tasks.push_event(database.get_bml_task_id(), bml_task::CONNECTION_CHANGE, new_event);
        LOG(TRACE) << "BML, sending hostap (" << hostap_mac
                   << ") active CONNECTION_CHANGE for IRE mac " << new_event->mac;
    }

    return result;
}

void son_actions::disconnect_client(db &database, ieee1905_1::CmduMessageTx &cmdu_tx,
                                    const std::string &client_mac, const std::string &bssid,
                                    eDisconnectType type, uint32_t reason,
                                    eClientDisconnectSource src)
{

    auto agent_mac = database.get_bss_parent_agent(tlvf::mac_from_string(bssid));

    auto request =
        message_com::create_vs_message<beerocks_message::cACTION_CONTROL_CLIENT_DISCONNECT_REQUEST>(
            cmdu_tx);

    if (request == nullptr) {
        LOG(ERROR) << "Failed building ACTION_CONTROL_CLIENT_DISCONNECT_REQUEST message!";
        return;
    }
    request->mac()    = tlvf::mac_from_string(client_mac);
    request->vap_id() = database.get_bss_vap_id(tlvf::mac_from_string(bssid));
    request->type()   = type;
    request->reason() = reason;
    request->src()    = src;

    const auto parent_radio = database.get_bss_parent_radio(bssid);
    son_actions::send_cmdu_to_agent(agent_mac, cmdu_tx, database, parent_radio);
    LOG(DEBUG) << "sending DISASSOCIATE request, client " << client_mac << " bssid " << bssid;
}void son_actions::send_cli_debug_message(db &database, ieee1905_1::CmduMessageTx &cmdu_tx,
                                         std::stringstream &ss)
{
    auto controller_ctx = database.get_controller_ctx();
    if (!controller_ctx) {
        LOG(ERROR) << "controller_ctx == nullptr";
        return;
    }

    auto response =
        message_com::create_vs_message<beerocks_message::cACTION_CLI_RESPONSE_STR>(cmdu_tx);

    if (response == nullptr) {
        LOG(ERROR) << "Failed building cACTION_CLI_RESPONSE_STR message!";
        return;
    }

    // In case we don't have enough space for node length, reserve 1 byte for '\0'
    size_t reserved_size =
        message_com::get_vs_cmdu_size_on_buffer<beerocks_message::cACTION_CLI_RESPONSE_STR>() - 1;

    auto buff_len = cmdu_tx.getMessageBuffLength();
    if (buff_len <= reserved_size) {
        LOG(ERROR) << "Invalid CMDU buffer length";
        return;
    }

    size_t max_size = buff_len - reserved_size;

    auto s      = ss.str();
    size_t size = std::min(s.size(), max_size);

    if (size > (std::numeric_limits<size_t>::max() - 1)) {
        LOG(ERROR) << "CLI response too large";
        return;
    }
    if (!response->alloc_buffer(size + 1)) {
        LOG(ERROR) << "Failed buffer allocation";
        return;
    }

    auto buf = response->buffer(0);
    if (!buf) {
        LOG(ERROR) << "Failed buffer allocation";
        return;
    }

    std::copy_n(s.data(), size, buf);
    buf[size] = 0;

    const int count = database.get_cli_sockets_count();
    for (int idx = 0; idx < count; idx++) {
        int fd = database.get_cli_socket_at(idx);
        if (beerocks::net::FileDescriptor::invalid_descriptor == fd) {
            break;
        }
        if (!controller_ctx->send_cmdu(fd, cmdu_tx)) {
            break;
        }
    }
}

void son_actions::handle_dead_radio(const sMacAddr &mac, bool reported_by_parent, db &database,
                                    task_pool &tasks)
{
    LOG(DEBUG) << "NOTICE: handling dead radio " << mac << " reported by parent "
               << reported_by_parent;

    std::shared_ptr<Agent::sRadio> radio = database.get_radio_by_uid(mac);
    if (!radio) {
        return;
    }

    std::string mac_str = tlvf::mac_to_string(mac);
    if (reported_by_parent) {
        database.set_radio_state(mac_str, beerocks::STATE_DISCONNECTED);
        set_radio_active(database, tasks, mac_str, false);

        /*
         * set all stations in the subtree as disconnected
         */
        int agent_monitoring_task_id = database.get_agent_monitoring_task_id();
        for (const auto &bss : radio->bsses) {
            for (const auto &client : bss.second->connected_stations) {
                // kill old roaming task
                int prev_task_id = client.second->roaming_task_id;
                if (tasks.is_task_running(prev_task_id)) {
                    tasks.kill_task(prev_task_id);
                }

                // ensure pointer passed to push_event remains valid
                auto mac_str_ptr = std::make_shared<std::string>(mac_str);
                tasks.push_event(agent_monitoring_task_id, STATE_DISCONNECTED, mac_str_ptr);

                // ensure pointer passed to push_event remains valid
                auto new_event = std::make_shared<bml_task::connection_change_event>();
                new_event->mac = tlvf::mac_to_string(client.first);
                tasks.push_event(database.get_bml_task_id(), bml_task::CONNECTION_CHANGE, new_event);

                LOG(DEBUG) << "BML, sending client disconnect CONNECTION_CHANGE for mac "
                           << new_event->mac;
            }
        }
    }
    LOG(DEBUG) << "handling dead radio, done for mac " << mac;
}

void son_actions::handle_dead_station(std::string mac, bool reported_by_parent, db &database,
                                      task_pool &tasks)
{
    LOG(DEBUG) << "NOTICE: handling dead station " << mac << " reported by parent "
               << reported_by_parent;

    // Copy required fields while holding the DB lock to avoid using station after unlock.
    bool is_wireless                  = false;
    bool is_bsta                      = false;
    bool handoff_flag                 = false;
    int association_handling_task_id  = -1;
    int steering_task_id              = -1;
    int btm_request_task_id           = -1;
    int roaming_task_id               = -1;
    sMacAddr station_al_mac           = {};

    {
        auto lock = database.lock();
        auto station = database.get_station(tlvf::mac_from_string(mac));
        if (!station) {
            return;
        }

        is_wireless                  = database.is_sta_wireless(mac);
        is_bsta                      = station->is_bSta();
        handoff_flag                 = database.get_sta_handoff_flag(*station);
        association_handling_task_id = station->association_handling_task_id;
        steering_task_id             = station->steering_task_id;
        btm_request_task_id          = station->btm_request_task_id;
        roaming_task_id              = station->roaming_task_id;
        station_al_mac               = station->al_mac;
    }

    if (is_wireless) {
        // If there is running association handling task already, terminate it.
        if (tasks.is_task_running(association_handling_task_id)) {
            tasks.kill_task(association_handling_task_id);
        }
    }

    if (reported_by_parent) {
        {
            auto lock = database.lock();
            database.set_sta_state(mac, beerocks::STATE_DISCONNECTED);
            database.set_sta_ipv4(mac, std::string());
        }

        // Notify steering task, if any, of disconnect.
        if (tasks.is_task_running(steering_task_id))
            tasks.push_event(steering_task_id, client_steering_task::STA_DISCONNECTED);

        // Notify btm_request task, if any, of disconnect.
        if (tasks.is_task_running(btm_request_task_id))
            tasks.push_event(btm_request_task_id, btm_request_task::STA_DISCONNECTED);

        if (handoff_flag) {
            LOG(DEBUG) << "handoff_flag == true, mac " << mac;
            // We're in the middle of steering, don't mark as disconnected (yet).
            return;
        } else {
            LOG(DEBUG) << "handoff_flag == false, mac " << mac;

            // If we're not in the middle of steering, kill roaming task
            if (tasks.is_task_running(roaming_task_id)) {
                tasks.kill_task(roaming_task_id);
            }
        }

        // If there is an instance of association handling task, kill it
        if (tasks.is_task_running(association_handling_task_id)) {
            tasks.kill_task(association_handling_task_id);
        }
    }

    // update bml listeners
    if (!is_bsta) {
        // ensure pointer passed to push_event remains valid
        auto new_event = std::make_shared<bml_task::connection_change_event>();
        new_event->mac = mac;
        tasks.push_event(database.get_bml_task_id(), bml_task::CONNECTION_CHANGE, new_event);

        LOG(DEBUG) << "BML, sending client disconnect CONNECTION_CHANGE for mac " << new_event->mac;
    } else {
        std::shared_ptr<agent> backhaul_bridge;
        {
            auto lock = database.lock();
            backhaul_bridge = database.m_agents.get(station_al_mac);
        }
        if (!backhaul_bridge) {
            LOG(ERROR) << "Station: " << mac << "does not have a bridge under it!";
        } else {
            // ensure pointer passed to push_event remains valid
            auto new_event = std::make_shared<bml_task::connection_change_event>();
            new_event->mac = tlvf::mac_to_string(backhaul_bridge->al_mac);
            LOG(DEBUG) << "BML, sending IRE disconnect CONNECTION_CHANGE for mac " << new_event->mac;
            tasks.push_event(database.get_bml_task_id(), bml_task::CONNECTION_CHANGE, new_event);
        }
    }

    LOG(DEBUG) << "handling dead station, done for mac " << mac;
}

bool son_actions::validate_beacon_measurement_report(beerocks_message::sBeaconResponse11k report,       const std::string &sta_mac,const std::string &bssid)
{
    if (report.rcpi > RCPI_MAX) {
        LOG(WARNING) << "RCPI Measurement is in reserved value range rcpi=" << report.rcpi;
    }

    return (report.rep_mode == 0) &&
           //      (report.rsni                                  >  0          ) &&
           (report.rcpi != RCPI_INVALID) &&
           //      (report.start_time                            >  0          ) &&
           //      (report.duration                              >  0          ) &&
           (report.channel > 0) && (tlvf::mac_to_string(report.sta_mac) == sta_mac) &&
           (tlvf::mac_to_string(report.bssid) == bssid);
}

/**
 * @brief Check if the operating classes of @a radio_basic_caps matches any of the operating classes
 *        in @a bss_info_conf
 */
/**
 * @param radio_basic_caps The AP Radio Basic Capabilities TLV of the radio
 * @param bss_info_conf The BSS Info we try to configure
 * @return true if one of the operating classes overlaps, false if they are disjoint
 */
bool son_actions::has_matching_operating_class(
    wfa_map::tlvApRadioBasicCapabilities &radio_basic_caps,
    const wireless_utils::sBssInfoConf &bss_info_conf)
{
    for (uint8_t i = 0; i < radio_basic_caps.operating_classes_info_list_length(); i++) {
        auto operating_class_info = std::get<1>(radio_basic_caps.operating_classes_info_list(i));
        for (auto operating_class : bss_info_conf.operating_class) {
            if (operating_class == operating_class_info.operating_class()) {
                return true;
            }
        }
    }
    return false;
}

bool son_actions::send_cmdu_to_agent(const sMacAddr &dest_mac, ieee1905_1::CmduMessageTx &cmdu_tx,
                                     db &database, const std::string &radio_mac)
{
    if (cmdu_tx.getMessageType() == ieee1905_1::eMessageType::VENDOR_SPECIFIC_MESSAGE) {
        if (!database.is_prplmesh(dest_mac)) {
            // skip non-prplmesh agents
            return false;
        }
        auto beerocks_header = message_com::get_beerocks_header(cmdu_tx);
        if (!beerocks_header) {
            LOG(ERROR) << "Failed getting beerocks_header!";
            return false;
        }

        beerocks_header->actionhdr()->radio_mac() = tlvf::mac_from_string(radio_mac);
        beerocks_header->actionhdr()->direction() = beerocks::BEEROCKS_DIRECTION_AGENT;
    }

    auto controller_ctx = database.get_controller_ctx();
    if (controller_ctx == nullptr) {
        LOG(ERROR) << "controller_ctx == nullptr";
        return false;
    }

    return controller_ctx->send_cmdu_to_broker(cmdu_tx, dest_mac, database.get_local_bridge_mac());
}

bool son_actions::send_ap_config_renew_msg(ieee1905_1::CmduMessageTx &cmdu_tx, db &database)
{
    // Create AP-Configuration renew message
    auto cmdu_header =
        cmdu_tx.create(0, ieee1905_1::eMessageType::AP_AUTOCONFIGURATION_RENEW_MESSAGE);
    if (!cmdu_header) {
        LOG(ERROR) << "Failed building IEEE1905 AP_AUTOCONFIGURATION_RENEW_MESSAGE";
        return false;
    }

    // Add MAC address TLV
    auto tlvAlMac = cmdu_tx.addClass<ieee1905_1::tlvAlMacAddress>();
    if (!tlvAlMac) {
        LOG(ERROR) << "Failed addClass ieee1905_1::tlvAlMacAddress";
        return false;
    }
    tlvAlMac->mac() = database.get_local_bridge_mac();

    // Add Supported-Role TLV
    auto tlvSupportedRole = cmdu_tx.addClass<ieee1905_1::tlvSupportedRole>();
    if (!tlvSupportedRole) {
        LOG(ERROR) << "Failed addClass ieee1905_1::tlvSupportedRole";
        return false;
    }
    tlvSupportedRole->value() = ieee1905_1::tlvSupportedRole::REGISTRAR;

    // Add Supported-Frequency-Band TLV
    auto tlvSupportedFreqBand = cmdu_tx.addClass<ieee1905_1::tlvSupportedFreqBand>();
    if (!tlvSupportedFreqBand) {
        LOG(ERROR) << "Failed addClass ieee1905_1::tlvSupportedFreqBand";
        return false;
    }
    // According to the Multi-AP Specification Version 2.0 section 7.1
    // Ragardless of what is sent here, the Agent will handle the Renew eitherway
    tlvSupportedFreqBand->value() = ieee1905_1::tlvSupportedFreqBand::eValue(0);

    LOG(INFO) << "Send AP_AUTOCONFIGURATION_RENEW_MESSAGE";
    return son_actions::send_cmdu_to_agent(network_utils::MULTICAST_1905_MAC_ADDR, cmdu_tx,
                                           database);
}

bool son_actions::send_topology_query_msg(const sMacAddr &dest_mac,
                                          ieee1905_1::CmduMessageTx &cmdu_tx, db &database)
{
    if (!cmdu_tx.create(0, ieee1905_1::eMessageType::TOPOLOGY_QUERY_MESSAGE)) {
        LOG(ERROR) << "Failed building TOPOLOGY_QUERY_MESSAGE message!";
        return false;
    }
    auto tlvProfile2MultiApProfile = cmdu_tx.addClass<wfa_map::tlvProfile2MultiApProfile>();
    if (!tlvProfile2MultiApProfile) {
        LOG(ERROR) << "addClass wfa_map::tlvProfile2MultiApProfile failed";
        return false;
    }
    return send_cmdu_to_agent(dest_mac, cmdu_tx, database);
}

bool son_actions::send_client_association_control(
    db &database, ieee1905_1::CmduMessageTx &cmdu_tx, const sMacAddr &agent_mac,
    const sMacAddr &agent_bssid, const std::unordered_set<sMacAddr> &station_list,
    const int &duration_sec,
    wfa_map::tlvClientAssociationControlRequest::eAssociationControl association_flag)
{
    if (!cmdu_tx.create(0, ieee1905_1::eMessageType::CLIENT_ASSOCIATION_CONTROL_REQUEST_MESSAGE)) {
        LOG(ERROR) << "Failed building CLIENT_ASSOCIATION_CONTROL_REQUEST_MESSAGE message";
        return false;
    }

    auto association_control_request_tlv =
        cmdu_tx.addClass<wfa_map::tlvClientAssociationControlRequest>();

    if (!association_control_request_tlv) {
        LOG(ERROR) << "addClass wfa_map::tlvClientAssociationControlRequest failed";
        return false;
    }

    association_control_request_tlv->bssid_to_block_client() = agent_bssid;
    association_control_request_tlv->association_control()   = association_flag;

    if (association_flag == wfa_map::tlvClientAssociationControlRequest::BLOCK ||
        association_flag == wfa_map::tlvClientAssociationControlRequest::TIMED_BLOCK) {
        association_control_request_tlv->validity_period_sec() = duration_sec;
    } else {
        association_control_request_tlv->validity_period_sec() = 0;
    }

    int index = 0;
    std::stringstream sta_list_str;
    for (auto station : station_list) {
        if (!association_control_request_tlv->alloc_sta_list()) {
            LOG(ERROR) << "can't alloc new station for client association control, currently allocd "
                       << index << " stations";
            if (association_control_request_tlv->sta_list_length() != index) {
                association_control_request_tlv->sta_list_length() = index;
            }
            return false;
        }
        auto sta_list_unblock         = association_control_request_tlv->sta_list(index);
        std::get<1>(sta_list_unblock) = station;
        index++;
        sta_list_str << tlvf::mac_to_string(station) << ", ";
    }

    std::string action_str =
        (association_flag == wfa_map::tlvClientAssociationControlRequest::BLOCK ||
         association_flag == wfa_map::tlvClientAssociationControlRequest::TIMED_BLOCK)
            ? "block"
            : "unblock";

    std::string duration_str =
        duration_sec != 0 ? "for " + std::to_string(duration_sec) + " seconds" : "";

    LOG(DEBUG) << "sending " << action_str << " request for " << sta_list_str.str() << " to agent "
               << tlvf::mac_to_string(agent_mac) << " for bssid "
               << association_control_request_tlv->bssid_to_block_client() << duration_str;
    return son_actions::send_cmdu_to_agent(agent_mac, cmdu_tx, database);
}

bool son_actions::handle_agent_ap_mld_configuration_tlv(db &database, const sMacAddr &al_mac,
                                                        ieee1905_1::CmduMessageRx &cmdu_rx)
{

    // Handle AgentApMldConfiguration TLV
    auto agent = database.m_agents.get(al_mac);
    if (!agent) {
        LOG(ERROR) << "Agent with mac is not found in database mac=" << al_mac;
        return false;
    }

    auto agent_ap_mld_configuration = cmdu_rx.getClass<wfa_map::tlvAgentApMldConfiguration>();
    if (!agent_ap_mld_configuration) {
        LOG(DEBUG) << "No tlvAgentApMldConfiguration TLV received";
    } else {
        // Update APMLD Database and Data Model based on received AgentApMldConfiguration TLV
        for (uint8_t ap_mld_it = 0; ap_mld_it < agent_ap_mld_configuration->num_ap_mld();
             ++ap_mld_it) {

            // Get AgentApMld configuration from TLV
            std::tuple<bool, wfa_map::cApMld &> ap_mld_tuple(
                agent_ap_mld_configuration->ap_mld(ap_mld_it));
            if (!std::get<0>(ap_mld_tuple)) {
                LOG(ERROR) << "Couldn't get AP MLD from tlvAgentApMldConfiguration";
                return false;
            }
            wfa_map::cApMld &ap_mld = std::get<1>(ap_mld_tuple);

            // SSID
            if (ap_mld.ssid_str().empty()) {
                // Dropping this MLD from updation, as SSID can't be empty for ApMld
                LOG(ERROR) << "SSID is empty in tlvAgentApMldConfiguration";
                continue;
            }

            // MAC
            if (!ap_mld.ap_mld_mac_addr_valid().is_valid) {
                // Dropping this MLD from updation, as MLD MAC is our key for ApMld
                // Hence, MLD MAC shall not be invalid
                LOG(ERROR) << "AP MLD MAC is not valid in tlvAgentApMldConfiguration";
                continue;
            }

            // Get or Allocate ApMld from DB
            auto mld_mac = ap_mld.ap_mld_mac_addr();
            Agent::sAPMLD *apmld = database.get_or_allocate_ap_mld(al_mac, mld_mac);
            if (!apmld) {
                LOG(ERROR) << "Failed to allocate/get AP MLD for al_mac=" << al_mac
                           << " mld_mac=" << mld_mac;
                continue;
            }

            // Update SSID
            apmld->mld_info.mld_ssid = ap_mld.ssid_str();

            // Update MLD MAC
            apmld->mld_info.mld_mac = ap_mld.ap_mld_mac_addr();

            // MLD MODE FLAGS - str, nstr, emlsr, emlmr
            apmld->mld_info.mld_mode = Agent::sMLDInfo::mode(0);
            if (ap_mld.modes().str) {
                apmld->mld_info.mld_mode =
                    Agent::sMLDInfo::mode(apmld->mld_info.mld_mode | Agent::sMLDInfo::mode::STR);
            }
            if (ap_mld.modes().nstr) {
                apmld->mld_info.mld_mode =
                    Agent::sMLDInfo::mode(apmld->mld_info.mld_mode | Agent::sMLDInfo::mode::NSTR);
            }
            if (ap_mld.modes().emlsr) {
                apmld->mld_info.mld_mode =
                    Agent::sMLDInfo::mode(apmld->mld_info.mld_mode | Agent::sMLDInfo::mode::EMLSR);
            }
            if (ap_mld.modes().emlmr) {
                apmld->mld_info.mld_mode =
                    Agent::sMLDInfo::mode(apmld->mld_info.mld_mode | Agent::sMLDInfo::mode::EMLMR);
            }

            for (uint8_t affiliated_ap_it = 0; affiliated_ap_it < ap_mld.num_affiliated_ap();
                 ++affiliated_ap_it) {
                std::tuple<bool, wfa_map::cAffiliatedAp &> affiliated_ap_tuple(
                    ap_mld.affiliated_ap(affiliated_ap_it));
                if (!std::get<0>(affiliated_ap_tuple)) {
                    LOG(ERROR) << "Couldn't get Affiliated AP from APMLD with SSID: "
                               << apmld->mld_info.mld_ssid
                               << " and with MLD MAC: " << apmld->mld_info.mld_mac;
                    return false;
                }

                // Get or Allocate Affiliated AP from DB
                auto ruid = std::get<1>(affiliated_ap_tuple).ruid();
                Agent::sAPMLD::sAffiliatedAP *affiliated_ap =
                    database.get_or_allocate_affiliated_ap(*apmld, ruid);
                if (!affiliated_ap) {
                    LOG(ERROR) << "Failed to allocate/get Affiliated AP for ruid=" << ruid;
                    continue;
                }

                // RUID
                // Update RUID for both existing and new Affiliated AP
                affiliated_ap->ruid = ruid;

                // BSSID
                if (std::get<1>(affiliated_ap_tuple)
                        .affiliated_ap_fields_valid()
                        .affiliated_ap_mac_addr_valid) {
                    affiliated_ap->bssid =
                        std::get<1>(affiliated_ap_tuple).affiliated_ap_mac_addr();
                } else {
                    affiliated_ap->bssid = beerocks::net::network_utils::ZERO_MAC;
                }

                // LinkID
                // Check existing TLVs while polulating valid
                if (std::get<1>(affiliated_ap_tuple).affiliated_ap_fields_valid().linkid_valid) {
                    affiliated_ap->link_id = std::get<1>(affiliated_ap_tuple).linkid();
                } else {
                    affiliated_ap->link_id = 0;
                }
            }
            // Add AP MLD Info in Data Model
            database.dm_add_ap_mld(al_mac, *apmld);
        }

        // Remove redundant APMLD/AffiliatedAP

        // Remove redundant ApMld entries without holding the mutex while calling into the data model.
        std::vector<sMacAddr> mlds_to_remove;
        {
            std::lock_guard<std::mutex> lock(agent->ap_mlds_mutex);
            for (const auto &kv : agent->ap_mlds) {
                const auto &mld_mac = kv.first;
                bool found          = false;

                for (uint8_t i = 0; i < agent_ap_mld_configuration->num_ap_mld(); ++i) {
                    auto ap_mld_tuple = agent_ap_mld_configuration->ap_mld(i);
                    if (std::get<0>(ap_mld_tuple)) {
                        auto &ap_mld = std::get<1>(ap_mld_tuple);
                        if (ap_mld.ap_mld_mac_addr_valid().is_valid &&
                            mld_mac == ap_mld.ap_mld_mac_addr()) {
                            found = true;
                            break;
                        }
                    }
                }

                if (!found) {
                    mlds_to_remove.push_back(mld_mac);
                }
            }
        }

        for (const auto &mld_mac : mlds_to_remove) {
            // Remove Database only if Data Model is removed
            if (database.dm_remove_ap_mld(agent->al_mac, mld_mac)) {
                {
                    std::lock_guard<std::mutex> lock(agent->ap_mlds_mutex);
                    agent->ap_mlds.erase(mld_mac);
                }
                LOG(DEBUG) << "Removed ApMld with MLD MAC: " << mld_mac
                           << " from Agent: " << al_mac;
            }
        }

        // Remove redundant Affiliated APs (added earlier, but not present now)
        std::vector<std::pair<sMacAddr, sMacAddr>> affiliated_to_remove; // (mld_mac, ruid)
        {
            std::lock_guard<std::mutex> lock(agent->ap_mlds_mutex);
            for (const auto &db_apmld_it : agent->ap_mlds) {
                const auto &mld_mac = db_apmld_it.first;

                // Find matching AP MLD in TLV
                bool tlv_apmld_found = false;
                uint8_t tlv_apmld_it = 0;
                for (; tlv_apmld_it < agent_ap_mld_configuration->num_ap_mld(); ++tlv_apmld_it) {
                    auto ap_mld_tuple = agent_ap_mld_configuration->ap_mld(tlv_apmld_it);
                    if (std::get<0>(ap_mld_tuple)) {
                        auto &ap_mld = std::get<1>(ap_mld_tuple);
                        if (ap_mld.ap_mld_mac_addr_valid().is_valid &&
                            mld_mac == ap_mld.ap_mld_mac_addr()) {
                            tlv_apmld_found = true;
                            break;
                        }
                    }
                }
                if (!tlv_apmld_found) {
                    continue; // handled by AP MLD removal above
                }

                auto ap_mld_tuple = agent_ap_mld_configuration->ap_mld(tlv_apmld_it);
                if (!std::get<0>(ap_mld_tuple)) {
                    continue;
                }
                auto &ap_mld = std::get<1>(ap_mld_tuple);

                for (const auto &db_affl_ap_it : db_apmld_it.second.affiliated_aps) {
                    const auto &ruid = db_affl_ap_it.first;
                    bool afflap_found = false;

                    for (uint8_t tlv_affl_ap_it = 0; tlv_affl_ap_it < ap_mld.num_affiliated_ap();
                         ++tlv_affl_ap_it) {
                        auto affiliated_ap_tuple = ap_mld.affiliated_ap(tlv_affl_ap_it);
                        if (std::get<0>(affiliated_ap_tuple)) {
                            if (ruid == std::get<1>(affiliated_ap_tuple).ruid()) {
                                afflap_found = true;
                                break;
                            }
                        }
                    }

                    if (!afflap_found) {
                        affiliated_to_remove.emplace_back(mld_mac, ruid);
                    }
                }
            }
        }

        for (const auto &entry : affiliated_to_remove) {
            const auto &mld_mac = entry.first;
            const auto &ruid    = entry.second;

            // Remove Database only if Data Model is removed
            if (database.dm_remove_affiliated_ap(agent->al_mac, mld_mac, ruid)) {
                {
                    std::lock_guard<std::mutex> lock(agent->ap_mlds_mutex);
                    auto it = agent->ap_mlds.find(mld_mac);
                    if (it != agent->ap_mlds.end()) {
                        it->second.affiliated_aps.erase(ruid);
                    }
                }
                LOG(DEBUG) << "Removed Affiliated AP with ruid: " << ruid
                           << " from ApMld with MLD MAC: " << mld_mac << " of Agent: " << al_mac;
            }
        }
    }
    return true;
}