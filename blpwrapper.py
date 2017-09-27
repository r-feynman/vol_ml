import blpapi
import numpy as np
import pandas as pd


def get_data(securities, fields='PX_LAST', **kwargs):

    # Check whether dataset is given as a string
    # (for a single dataset) or an array (for a multiset call)
    if isinstance(securities, string_types):
        securities = list(securities)
    # Array
    elif isinstance(securities, list):
        pass
    else:
        raise InvalidRequestError(Message.ERROR_DATASET_FORMAT)
        exit()

    if isinstance(fields, string_types):
        fields = list(fields)
    # Array
    elif isinstance(fields, list):
        pass
    else:
        raise InvalidRequestError(Message.ERROR_DATASET_FORMAT)
        exit()

    session = blpapi.Session()
    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")
        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        # Append Securities
        for sec in securities:
            request.getElement("securities").appendValue(sec)

        # Append Fields
        for field in fields:
            request.getElement("fields").appendValue(field)

        # Set the various parameters
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "MONTHLY")
        request.set("startDate", "20060101")
        request.set("endDate", "20061231")
        request.set("maxDataPoints", 100)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)

        BigTable = []

        # Process received events
        while True:
            ev = session.nextEvent(500)
            if ev.eventType() == blpapi.Event.PARTIAL_RESPONSE or ev.eventType() == blpapi.Event.RESPONSE:
                for msg in ev:
                    security_data = msg.getElement("securityData")
                    field_data_array = security_data.getElement("fieldData")
                    field_data_list = [field for field in field_data_array.values()]
                    security_name = security_data.getElement("security").getValue()
                    sec_list.append(security_name)

                    sec_table = []

                    for element in field_data_list:
                        sec_table.append(element.getElement("date").getValue())
                        for field in fields:
                            sec_table.append(element.getElement(field).getValue())

                    big_table.append(sec_table)

            if ev.eventType() == blpapi.Event.RESPONSE:
                break

            return_dict = {}
            for security, name in zip(big_table, sec_list):
                df = pd.DataFrame(columns=['Date'] + fields, data=np.array(security).reshape(-1, len(fields) + 1))
                df.set_index('Date', inplace=True, drop=True)
                return_dict[name] = df.copy()

            if len(return_dict) == 1:
                return return_dict.popitem()
            else:
                return return_dict
    finally:
        # Stop the session
        session.stop()
