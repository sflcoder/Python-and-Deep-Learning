from datetime import datetime
from datetime import timedelta


class person():
    def __init__(self, name, age, gender, phoneNumber):
        self.name = name
        self.age = age
        self.gender = gender
        self.phoneNumber = phoneNumber

    # Redefine the print function for person class
    def __str__(self):
        return "\nPerson information:\nName:\t" + self.name + "\tAge:\t" + str(
            self.age) + "\tGender:\t" + self.gender + "\tPhone:\t" + str(
            self.phoneNumber)


# employee class inherits person class
class employee(person):
    def __init__(self, name, age, gender, phoneNumber, employer, departure, arrival, employeeID):
        super(employee, self).__init__(name, age, gender, phoneNumber)
        self.departure = departure
        self.arrival = arrival
        self.employer = employer
        self.employeeID = employeeID

    def __str__(self):
        return "\nEmployee information:\nName:\t" + self.name + "\tAge:\t" + str(
            self.age) + "\tGender:\t" + self.gender + "\tPhone:\t" \
               + str(self.phoneNumber) + "\tdeparture:\t" + str(self.departure) + "\tarrival:\t" + str(self.arrival) \
               + "\temployer:\t" + str(self.employer)


# passenger class inherits person class
class passenger(person):
    def __init__(self, name, age, gender, phoneNumber, departure, arrival):
        super(passenger, self).__init__(name, age, gender, phoneNumber)
        self.departure = departure
        self.arrival = arrival

    def __str__(self):
        return "\nPassenger information:\nName:\t" + self.name + "\tAge:\t" + str(
            self.age) + "\tGender:\t" + self.gender + "\tPhone:\t" \
               + str(self.phoneNumber) + "\tdeparture:\t" + str(self.departure) + "\tarrival:\t" + str(self.arrival)


class flight():
    def __init__(self, flightNum, departure, arrival, flightTime):
        self.flightNum = flightNum
        self.departure = departure
        self.arrival = arrival
        self.flightTime = flightTime
        self.seatsCapacity = 100
        self.seatNum = 0

    def __str__(self):
        return "FlightNum:\t" + self.flightNum + "\tDeparture:\t" + str(self.departure) + "\tarrival:\t" \
               + self.arrival + "\tTime:\t" + self.flightTime

    # Allocate seat for passenger
    def allocateSeatNum(self):
        if self.seatNum < self.seatsCapacity:
            self.seatNum += 1
            print('Your seat number is:', self.seatNum)

    # display the boarding time for passenger
    def getBoardingtime(self):
        flightTime = datetime.strptime(self.flightTime, "%Y-%m-%d %H:%M:%S")
        boardingTime = flightTime + timedelta(hours=-1)
        print('Boarding Time: ', boardingTime)


class airline():
    def __init__(self, airlineName):
        self.airlineName = airlineName
        self.flightNum = 0
        self.employeeNum = 0
        self.flightList = []
        self.employees = []

    # Add employee to the employee list of airline
    def addEmployee(self, employee):
        self.employees.append(employee)
        self.employeeNum += 1

    # add flight to the flight list of airline
    def addFlight(self, flight):
        self.flightList.append(flight)
        self.employeeNum += 1

    # Display the flight information
    def flightsInfor(self):
        for flight in self.flightList:
            print(flight)

    # Book ticket for passenger
    def ticket(self, passenger):
        print('\n====               Welcome to ', self.airlineName, '               ====')
        print('====  Booking ticket from', passenger.departure, ' to', passenger.arrival, '  ====\n')
        for flight in self.flightList:
            if passenger.departure == flight.departure and passenger.arrival == flight.arrival:
                print('Congratulations', passenger.name, '\nYour Flight Number: ', self.airlineName, flight.flightNum)
                flight.allocateSeatNum()
                flight.getBoardingtime()
                break
        print('Sorry: There is no flight available from', passenger.departure, ' to', passenger.arrival, 'now.')


person1 = person("Mike", 25, "Male", 12345678)
print(person1)

employee1 = employee("Jack", 40, "Male", 8165555555, 'Space X', 'MCI', 'SFO', 1)
print(employee1)

passenger1 = passenger("Jack", 40, "Male", 8165555555, 'Kansas City', 'San Francisco')
passenger2 = passenger("Kevin", 20, "Male", 9136666666, 'Kansas City', 'New York')
print(passenger1)
print(passenger2)

flightTime1 = '2017-11-24 17:30:00'
flightTime2 = '2017-11-24 15:00:00'
flightMCItoSFO = flight("1234", "Kansas City", "San Francisco", flightTime1)
flightMCItoLAX = flight("5678", "Kansas City", "Los Angeles", flightTime2)

spaceAir = airline('Space Air')
spaceAir.addFlight(flightMCItoSFO)
spaceAir.addFlight(flightMCItoLAX)
print('\nThe flight of Space Air:')
print(spaceAir.flightsInfor())

spaceAir.ticket(passenger1)
spaceAir.ticket(passenger2)
