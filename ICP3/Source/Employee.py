class Employee:
    # Total number of Employees
    employeeNum = 0
    # Total salary of all Employees
    salarySum = 0

    def __init__(self, n, f, s, d ):
        self.name = n
        self.family = f
        self.salary = s
        self.department = d
        Employee.employeeNum += 1
        Employee.salarySum += s
    # Calculate the average salary of all Employees
    def aveSalary(self):
        print('The average salary is: ',Employee.salarySum / Employee.employeeNum)
    # Define the print function for class Employee
    def __str__(self):
        return 'Name: '+ self.name+ ' ' + self.family + '   Salary: '+ str(self.salary)+'    Department: '+self.department

class FulltimeEmployee(Employee):
    def __init__(self, n, f, s, d , b):
        Employee.__init__(self, n, f, s, d)
        self.isFulltime = b

    # Define the print function for class FulltimeEmployee
    def __str__(self):
        return 'Name: '+ self.name+ ' ' + self.family + '   Salary: '+ str(self.salary)+'    Department: '+self.department + '   Full time: ' + str(self.isFulltime)

ftEmployee = FulltimeEmployee('Michale', 'Li', 3000, 'CS', True)
print(ftEmployee)
ftEmployee.aveSalary()

employee = Employee('Peter', 'Li', 2000, 'EE' )
print(employee)
employee.aveSalary()





