$(document).ready(function(e) {

    /// Takes care of initialization for current page
    function txml2html() {
        // make sure the map is scaled along with the image
        $('img[usemap]').rwdImageMaps();

        initNodeDialog();

        // handlers for clicking on image features
        $('area').on('click', handleAreaClick);

    }

    /// extract data from a node url
    function url2hashes(href) {
        result = {'var': null};
        if (href.indexOf('#n') === 0) {
            // this is an opperation
            result['node'] = href.substring(2);
            result['mode'] = 'node';
        } else if (href.indexOf('#inputn') === 0) {
            // this is an input variable
            hash_node = href.substring(7);
            split_loc = hash_node.indexOf('x');
            result['var'] = hash_node.substring(split_loc+1);
            result['node'] = hash_node.substring(0,split_loc);
            result['mode'] = 'input';
        } else if (href.indexOf('#outputn') === 0) {
            // this is an output variable
            hash_node = href.substring(8);
            split_loc = hash_node.indexOf('x');
            result['var'] = hash_node.substring(split_loc+1);
            result['node'] = hash_node.substring(0,split_loc);
            result['mode'] = 'output';
        } else {
            result['mode'] = 'unknown';
        }
        return result;
    }

    /// Produce the html to represent variable choice for a list of variables
    function varsButtons(inplst, label_prefix, var_prefix) {
        var btn_template = '<input type="checkbox" id="%id%" name="%name%" %chk%><label id="lbl%idlbl%" for="%lname%">%label%</label>';
        new_html = '';
        in_cnt = inplst.length;
        for (i = 0; i < in_cnt; ++i) {
            varinst = inplst[i];
            lbl = var_prefix + i;
            varhash = varinst['hash'];
            varname = label_prefix + varhash;
            checked = '';
            new_html = new_html + btn_template
                .replace('%id%', varname).replace('%name%', varname)
                .replace('%chk%', checked).replace('%idlbl%', varname)
                .replace('%lname%', varname).replace('%label%', lbl);
        }
        return new_html;
    }

    /// Create the interface for a variable
    function varKindScalar(varinst) {
        $('#ndd_var_value').html('<br><br><center><strong>Value = ' + 
            varinst['data'] + '</strong></center>');
    }

    /// 
    function defaultDataTable() {
        var data = {
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return '<center><strong>' + data + '</strong></denter>';
                },
                "targets": 0
            }],
            searching: false,
            ordering: false,
            paging: false
        };
        return data;
    }

    /// Create columns headers
    function varHeader(cnt) {
        var arr = [{ "title": 'x', "class": "center" }];
        for (i = 0; i < cnt; ++i) {
            arr.push({ "title": String(i+1), "class": "center" });
        }
        return arr;
    }

    /// Create the interface for a variable
    function varKindVector(varinst) {
        var data = [1];
        data = data.concat(varinst['data']);
        var headers = varHeader(varinst['shape'][0]);
        $('#ndd_table').dataTable( $.extend({}, defaultDataTable(), {
            "data": [data],
            "columns": headers
        }));

    }

    /// Create the interface for a variable
    function varKindMatrix(varinst) {
        var data = [];
        var dlen = varinst['data'].length;
        for (i = 0; i < dlen; ++i) {
            arline = [String(i+1)];
            data.push(arline.concat(varinst['data'][i]));
        }
        var headers = varHeader(varinst['shape'][1]);
        $('#ndd_table').dataTable( $.extend({}, defaultDataTable(), {
            "data": data,
            "columns": headers
        }));
    }

    /// Create the interface for a variable
    function varKindNDArray(varinst) {
        // TODO: implement
        var data = [];
        var dlen = varinst['data'].length;
        for (i = 0; i < dlen; ++i) {
            arline = [String(i+1)];
            data.push(arline.concat(varinst['data'][i]));
        }
        var headers = varHeader(varinst['shape'][1]);
        $('#ndd_table').dataTable( $.extend({}, defaultDataTable(), {
            "data": data,
            "columns": headers
        }));
    }

    /// Create the interface for a variable
    function varByKind(varinst) {
        $('#ndd_var_value').html('');
        if (varinst['kind'] === 'integer') {
            varKindScalar(varinst);
        } else if (varinst['kind'] === 'real') {
            varKindScalar(varinst);
        } else if (varinst['kind'] === 'cplx') {
            varKindScalar(varinst);
        } else {
            $('#ndd_var_value').html('<table id="ndd_table" class="display"></table>');
            if (varinst['kind'] === 'vector') {
                varKindVector(varinst);
            } else if (varinst['kind'] === 'matrix') {
                varKindMatrix(varinst);
            } else if (varinst['kind'] === 'ndarray') {
                varKindNDArray(varinst);
            } else {
                console.log('Unknown variable kind ' + varinst['kind']);
            }
        }
    }

    /// Create the interface for a variable
    function showVariable(varname) {
        var dialog = $("#nodedialog");
        // the hash for this variable
        var varhash = varname.substring(4);
        // the type (vari, varo)
        var vartype = varname.substring(0, 4);

        // get current epoch and node
        var epoch = dialog.data('epoch');
        var hash_node = dialog.data('hash_node');

        // the data for this node
        var opnode = window.tx2h_data[hash_node];
        var varlst = null;
        if (vartype === 'vari') {
            varlst = opnode['epochs'][epoch]['input'];
        } else {
            varlst = opnode['epochs'][epoch]['output'];
        }
        // find our variable in the list of this node
        for (i = 0; i < varlst.length; ++i) {
            if (varlst[i]['hash'] == varhash) {
                // found it
                varinst = varlst[i];
                shape = varinst['shape'];
                if (shape.length > 0) {
                    shape = ' and shape <strong>' + shape.join(' x ') + '</strong>';
                }
                $('#ndd_var_brief').html(
                    varinst['name'] + ' of type <strong>' +
                    varinst['dtype'] + '</strong>' + shape);
                varByKind(varinst);
                break;
            }
        }

        // mark this as being current variable
        $("#ndd_form_inner").data('var', varname);
    }

    /// Clicking on one of the check boxes representing variables
    function varButtonClick(event) {
        var dialog = $("#nodedialog");
        var checkbox = $(this);
        // full name for the variable
        var varname = checkbox.attr('name');

        // see if this is the current variable
        if ($("#ndd_form_inner").data('var') == varname) {
            event.preventDefault();
            return;
        }

        // uncheck all buttons except ours
        btns = $("#ndd_form_inner > input[type='checkbox']");
        for (i = 0; i < btns.length; ++i) {
            if (this != btns[i]) {
                btn = $(btns[i]);
                btn.removeAttr('checked');
                btn.button('refresh');
            }
        }
        $("#ndd_form_inner").buttonset("refresh");

        showVariable(varname);
    }

    /// Produce the html to represent variable choice for a list of variables
    function allVarsButtons(opnode) {
        var epoch = $("#nodedialog").data('epoch');
        inplst = opnode['epochs'][epoch]['input'];
        new_html = varsButtons(inplst, 'vari', 'Input ');
        outlst = opnode['epochs'][epoch]['output'];
        new_html = new_html + varsButtons(outlst, 'varo', 'Output ');
        $("#ndd_form").html('<div id="ndd_form_inner">' + new_html + '</div>');
        $("#ndd_form_inner").buttonset();

        btns = $("#ndd_form_inner > input[type='checkbox']");
        for (i = 0; i < btns.length; ++i) {
            btn = $(btns[i]);
            btn.button();
        }
        $("#ndd_form_inner > input[type='checkbox']").on("click", varButtonClick);
    }

    /// Show the dialog and fill in node related content
    function showNodeDialog(node, vartag) {
        var dialog = $("#nodedialog");
        dialog = dialog.dialog({
            title: 'Op ' + node});
        dialog.data('hash_node', node);
        dialog.data('epoch', 0);

        // extract info about this node in symbolic vars
        if (node in window.tx2h_data) {
            opnode = window.tx2h_data[node];
        } else {
            alert("In Full mode only Op nodes show information");
            return;
        }
        name = opnode['name'];
        in_cnt = opnode['inputs'];
        out_cnt = opnode['outputs'];
        op = opnode['op'];
        ep_cnt = opnode['epochs'].length;

        $("#ndd_node_name").text(name);
        $("#ndd_node_op").text(op);
        $("#ndd_node_inp").text(in_cnt);
        $("#ndd_node_outp").text(out_cnt);
        $("#ndd_node_ep").text(ep_cnt);

        // create buttons for variables
        allVarsButtons(opnode);

        // show variable
        $('#ndd_var_brief').html('');
        $('#ndd_var_value').html('');
        if (vartag !== null) {
            showVariable(vartag);
        }

        dialog.dialog('open');

        return dialog;
    }

    /// respond to clicks inside the image
    function handleAreaClick(event) {
        event.preventDefault();

        // get user selection
        selection = url2hashes($(this).attr('href'));
        if (selection['mode'] === 'unknown') {
            alert('Unknown selection: ' + $(this).attr('href'));
            return;
        }

        // see if we need to show a variable
        var vartag = null;
        if (selection['mode'] === 'input') {
            vartag = 'vari' + selection['var'];
        } else if (selection['mode'] === 'output') {
            vartag = 'varo' + selection['var'];
        }

        // show node data
        dialog = showNodeDialog(selection['node'], vartag);
    }

    /// prepare the dialog with detailed information about the node
    function initNodeDialog() {
        try {
            $("#nodedialog").dialog({
                autoOpen: false,
                show: {
                    effect: "explode",
                    duration: 200
                },
                hide: {
                    effect: "explode",
                    duration: 200
                },
                height: $(window).height() / 2,
                width: $(window).width() / 2,
                create: function () {
                    $("#nodedialog").parent().children('.ui-dialog-titlebar')
                        .prepend('<button id="pindown"' + 
                                 ' class="ui-button ui-widget ' + 
                                 ' ui-state-default ui-corner-all ' +
                                 ' ui-button-icon-only PinDialog" ' +
                                 ' role="button" aria-disabled="false" ' + 
                                 ' title="Pin down">' + 
                                 '<span class="ui-button-icon-primary ' +
                                 ' ui-icon ui-icon-pin-w">' + 
                                 '</span></button>');
                }
            });
        } catch(err) {
            console.log(err);
        }
    }

    // initialize the page
    txml2html();


});
