/* SPDX-License-Identifier: BSD-2-Clause-Patent
 *
 * SPDX-FileCopyrightText: 2022 the prplMesh contributors (see AUTHORS.md)
 *
 * This code is subject to the terms of the BSD+Patent license.
 * See LICENSE file for more details.
 */

#ifndef PRPLMESH_AMX_CLIENT_H
#define PRPLMESH_AMX_CLIENT_H

#include <amxc/amxc.h>
#include <amxp/amxp.h>

#include <amxd/amxd_action.h>
#include <amxd/amxd_dm.h>
#include <amxd/amxd_object.h>
#include <amxd/amxd_object_event.h>
#include <amxd/amxd_transaction.h>

#include <amxb/amxb.h>
#include <amxb/amxb_register.h>

#include <amxo/amxo.h>
#include <amxo/amxo_save.h>

#include <easylogging++.h>

#include <iostream>
#include <locale.h>
#include <time.h>

#include <memory>
#include <string>

namespace beerocks {
namespace prplmesh_amx {

class AmxClient {

public:
    AmxClient() = default;
    AmxClient(const AmxClient &)            = delete;
    AmxClient &operator=(const AmxClient &) = delete;
    ~AmxClient();

    // Connect to an ambiorix.
    bool amx_initialize(const std::string &amxb_backend, const std::string &bus_uri);

    // Get an object from bus using object_path.
    using amxc_var_ptr = std::unique_ptr<amxc_var_t, void (*)(amxc_var_t *)>;

    static void amxc_var_deleter(amxc_var_t *v)
    {
        if (v) {
            amxc_var_delete(&v);
        }
    }

    amxc_var_ptr get_object(const std::string &object_path)
    {
        bool dummy = false;
        return get_object(object_path, dummy);
    }

    amxc_var_ptr get_object(const std::string &object_path, bool &request_timed_out);

    const amxc_htable_t *get_htable_object(const std::string &object_path);

    /**
     * @returns status from amxb_error.h (AMXB_STATUS_OK on success etc.)
     */
    int set_object(const std::string &path, amxc_var_t *value, amxc_var_t *ret = nullptr);

private:
    amxb_bus_ctx_t *bus_ctx = nullptr;
};

} // namespace prplmesh_amx
} // namespace beerocks

#endif // PRPLMESH_AMX_CLIENT_H