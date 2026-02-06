/* SPDX-License-Identifier: BSD-2-Clause-Patent
 *
 * SPDX-FileCopyrightText: 2019-2020 the prplMesh contributors (see AUTHORS.md)
 *
 * This code is subject to the terms of the BSD+Patent license.
 * See LICENSE file for more details.
 */

#ifndef __CONTROLLER_UCC_LISTENER_H__
#define __CONTROLLER_UCC_LISTENER_H__

#include <bcl/beerocks_ucc_listener.h>

#include <bcl/beerocks_ucc_server.h>

#include "db/db.h"

using namespace son;

using radio_preference_tlv_format = std::map<std::pair<uint8_t, uint8_t>, std::set<uint8_t>>;

namespace beerocks {
class controller_ucc_listener : public beerocks_ucc_listener {
public:
    controller_ucc_listener(db &database, ieee1905_1::CmduMessageTx &cmdu_tx,
                            std::unique_ptr<beerocks::UccServer> ucc_server);
    ~controller_ucc_listener() override;

private:
    std::string fill_version_reply_string() override;
    bool clear_configuration();
    bool clear_configuration(const sMacAddr &al_mac);
    bool send_cmdu_to_destination(ieee1905_1::CmduMessageTx &cmdu_tx,
                                  const std::string &dest_mac = std::string()) override;
    bool handle_start_wps_registration(const std::string &band, std::string &err_string) override;
    bool handle_dev_get_param(std::unordered_map<std::string, std::string> &params,
                              std::string &value) override;
    bool handle_dev_set_rfeature(const std::unordered_map<std::string, std::string> &params,
                                 std::string &err_string) override;
    bool handle_dev_exec_action(const std::unordered_map<std::string, std::string> &params,
                                std::string &err_string) override;
    bool handle_custom_command(const std::unordered_map<std::string, std::string> &params,
                               std::string &err_string) override;

    /**
     * @brief Handle spesific DEV_SEND_1905 that require internal workflow logic (like tasks)
     *
     * @param[in] params Command parameters.
     * @param[in] m_id Message id.
     * 
     * @return true if case was handled and false if case was not handled and general handling is required.
     */
    bool handle_dev_send_1905_internally(const std::unordered_map<std::string, std::string> &params,
                                         uint16_t m_id) override;

    static std::string parse_bss_info(const std::string &bss_info_str,
                                      son::wireless_utils::sBssInfoConf &bss_info_conf,
                                      std::string &err_string);

    db &m_database;
    std::unordered_set<sMacAddr> m_bss_info_cleared_mac;

private:
    /**
     * @brief Callback handler function for "dev_reset_default" WFA-CA command.
     *
     * @param[in] fd File descriptor of the socket connection the command was received through.
     * @param[in] params Command parameters.
     */
    void handle_dev_reset_default(int fd,
                                  const std::unordered_map<std::string, std::string> &params);

    /**
     * @brief Callback handler function for "dev_set_config" WFA-CA command.
     *
     * @param[in] params Command parameters.
     * @param[out] err_string Contains an error description if the function fails.
     * 
     * @return true on success and false otherwise.
     */
    bool handle_dev_set_config(const std::unordered_map<std::string, std::string> &params,
                               std::string &err_string);

    /**
     * @brief Converts the radio preference map to TLV format.
     *
     * This function takes a map containing radio preference data and converts it 
     * into a format suitable for TLV encoding.
     *
     * @param[in] radio_preference The map of radio preferences to be converted.
     * 
     * @return A map formatted for TLV encoding.
     */
    radio_preference_tlv_format convert_preference_report_map_to_tlv_format(
        const Agent::sRadio::PreferenceReportMap &radio_preference);

    /**
     * @brief Creates a TLV Channel Preference object.
     *
     * This function constructs and populates a TLV Channel Preference object using 
     * the given radio MAC address and formatted radio preference data.
     *
     * @param[in] radio_mac The MAC address of the radio.
     * @param[in] formatted_preference The formatted radio preference data in TLV format.
     * 
     * @return true if the TLV Channel Preference object is successfully created; false otherwise.
     */
    bool create_channel_preference_tlv(const sMacAddr &radio_mac,
                                       const radio_preference_tlv_format &formatted_preference);
};

} // namespace beerocks

#endif // __CONTROLLER_UCC_LISTENER_H__
